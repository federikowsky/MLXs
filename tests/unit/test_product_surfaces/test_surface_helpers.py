"""Pure helper tests for Layer 4 product surfaces."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from mlxs._errors import CapacityExceededError, RequestTimeoutError
from mlxs._types import FinishReason, GenerateOptions, TokenEvent
from mlxs.config.schema import AppConfig
from mlxs.observability.metrics import InMemoryMetrics, NoOpMetrics
from mlxs.protocols.prompt_cache import PromptCacheStats
from mlxs.product_surfaces.compat_openai import (
    _execute_generation,
    _stream_generation,
    build_openai_options,
)
from mlxs.product_surfaces.http import create_app
from mlxs.product_surfaces.lifecycle import RuntimeLifecycle
from mlxs.product_surfaces.observability import metrics_snapshot
from mlxs.server.queue import RequestQueue
from starlette.testclient import TestClient


def _runtime(
    *,
    metrics_enabled: bool = False,
    request_timeout: float = 300.0,
    max_queue_size: int = 64,
    max_concurrent_requests: int = 16,
    generate_fn=None,
):
    config = AppConfig()
    config = config.model_copy(
        update={
            "server": config.server.model_copy(
                update={
                    "request_timeout": request_timeout,
                    "max_queue_size": max_queue_size,
                    "max_concurrent_requests": max_concurrent_requests,
                }
            ),
            "observability": config.observability.model_copy(
                update={"metrics_enabled": metrics_enabled}
            ),
        }
    )

    def _default_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return iter([TokenEvent(token_id=1, text="ok", finish_reason=FinishReason.STOP)])

    return SimpleNamespace(
        config=config,
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        prompt_cache=SimpleNamespace(
            stats=lambda: PromptCacheStats(
                hit_count=0,
                miss_count=0,
                eviction_count=0,
                entry_count=0,
                total_bytes=0,
            )
        ),
        metrics=InMemoryMetrics() if metrics_enabled else NoOpMetrics(),
        generate_fn=generate_fn or _default_generate,
        batch_host=None,
        request_queue=RequestQueue(
            max_size=max_queue_size,
            max_concurrent=max_concurrent_requests,
            timeout=request_timeout,
        ),
    )


def test_build_openai_options_layer4_mapping() -> None:
    options = build_openai_options(
        {
            "max_tokens": 32,
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["END"],
            "stream": True,
            "logprobs": True,
            "top_logprobs": 3,
        }
    )
    assert isinstance(options, GenerateOptions)
    assert options.max_tokens == 32
    assert options.stop_sequences == ("END",)
    assert options.logprobs is True
    assert options.top_logprobs == 3


def test_lifecycle_payload_reports_layer4_status() -> None:
    lifecycle = RuntimeLifecycle(model_id="m")
    lifecycle.mark_ready()
    payload = lifecycle.health_payload(metrics_enabled=True, metrics_route_enabled=True)
    assert payload["status"] == "ok"
    assert payload["ready"] is True
    assert payload["model_id"] == "m"


def test_metrics_snapshot_main_app_mode() -> None:
    runtime = _runtime(metrics_enabled=True)
    runtime.metrics.counter("http_requests_total", 1.0)
    snapshot = metrics_snapshot(runtime)
    assert snapshot["enabled"] is True
    assert snapshot["route_mode"] == "main_app"
    assert snapshot["metrics_port_compatibility_only"] is True
    assert snapshot["backend"] == "in_memory"
    assert snapshot["prompt_cache"]["available"] is True
    assert snapshot["prompt_cache"]["entry_count"] == 0


def test_metrics_snapshot_includes_prompt_cache_stats_and_limits() -> None:
    runtime = _runtime(metrics_enabled=True)
    runtime.prompt_cache = SimpleNamespace(
        stats=lambda: PromptCacheStats(
            hit_count=3,
            miss_count=1,
            eviction_count=2,
            entry_count=4,
            total_bytes=123456,
        )
    )

    snapshot = metrics_snapshot(runtime)

    prompt_cache = snapshot["prompt_cache"]
    assert prompt_cache["enabled"] is True
    assert prompt_cache["max_entries"] == 100
    assert prompt_cache["hit_count"] == 3
    assert prompt_cache["miss_count"] == 1
    assert prompt_cache["lookup_count"] == 4
    assert prompt_cache["hit_ratio"] == 0.75
    assert prompt_cache["eviction_count"] == 2
    assert prompt_cache["entry_count"] == 4
    assert prompt_cache["total_bytes"] == 123456


def test_metrics_endpoint_exposes_prompt_cache_section() -> None:
    runtime = _runtime(metrics_enabled=True)
    runtime.prompt_cache = SimpleNamespace(
        stats=lambda: PromptCacheStats(
            hit_count=2,
            miss_count=3,
            eviction_count=1,
            entry_count=4,
            total_bytes=987654,
        )
    )

    client = TestClient(create_app(runtime))
    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt_cache"]["enabled"] is True
    assert payload["prompt_cache"]["available"] is True
    assert payload["prompt_cache"]["hit_count"] == 2
    assert payload["prompt_cache"]["miss_count"] == 3
    assert payload["prompt_cache"]["lookup_count"] == 5
    assert payload["prompt_cache"]["hit_ratio"] == 0.4
    assert payload["prompt_cache"]["eviction_count"] == 1
    assert payload["prompt_cache"]["entry_count"] == 4
    assert payload["prompt_cache"]["total_bytes"] == 987654


def test_execute_generation_rejects_when_queue_full() -> None:
    runtime = _runtime(max_queue_size=1, max_concurrent_requests=1)

    async def _run() -> None:
        await runtime.request_queue.put({"id": "active"})
        task = asyncio.create_task(runtime.request_queue.put({"id": "pending"}, timeout=0.1))
        await asyncio.sleep(0)
        with pytest.raises(CapacityExceededError, match="queue full"):
            await _execute_generation(
                runtime,
                "prompt",
                GenerateOptions(),
                input_embeddings=None,
            )
        admitted = await runtime.request_queue.get()
        assert admitted == {"id": "pending"}
        await task
        await runtime.request_queue.get()

    asyncio.run(_run())


def test_execute_generation_times_out_at_layer4_surface() -> None:
    def slow_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        time.sleep(0.05)
        return iter([TokenEvent(token_id=1, text="late", finish_reason=FinishReason.STOP)])

    runtime = _runtime(request_timeout=0.01, generate_fn=slow_generate)

    with pytest.raises(RequestTimeoutError, match="timed out"):
        asyncio.run(
            _execute_generation(
                runtime,
                "prompt",
                GenerateOptions(),
                input_embeddings=None,
            )
        )


def test_execute_generation_times_out_while_waiting_for_admission() -> None:
    runtime = _runtime(
        request_timeout=0.01,
        max_queue_size=1,
        max_concurrent_requests=1,
    )

    async def _run() -> None:
        await runtime.request_queue.put({"id": "active"})
        with pytest.raises(RequestTimeoutError, match="timed out"):
            await _execute_generation(
                runtime,
                "prompt",
                GenerateOptions(),
                input_embeddings=None,
            )
        assert runtime.request_queue.pending_count == 0
        await runtime.request_queue.get()

    asyncio.run(_run())


def test_execute_generation_uses_batch_host_for_text_only_requests() -> None:
    called: list[tuple[str, GenerateOptions]] = []

    class _BatchHost:
        async def execute(self, prompt: str, options: GenerateOptions) -> list[TokenEvent]:
            called.append((prompt, options))
            return [TokenEvent(token_id=1, text="ok", finish_reason=FinishReason.STOP)]

    runtime = _runtime()
    runtime.batch_host = _BatchHost()

    events = asyncio.run(
        _execute_generation(
            runtime,
            "prompt",
            GenerateOptions(),
            input_embeddings=None,
        )
    )

    assert called
    assert [event.text for event in events] == ["ok"]


def test_execute_generation_falls_back_when_input_embeddings_are_present() -> None:
    called = {"batch": 0, "generate": 0}

    class _BatchHost:
        async def execute(self, prompt: str, options: GenerateOptions) -> list[TokenEvent]:
            del prompt, options
            called["batch"] += 1
            return [TokenEvent(token_id=1, text="batched", finish_reason=FinishReason.STOP)]

    def _generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        called["generate"] += 1
        return iter([TokenEvent(token_id=1, text="fallback", finish_reason=FinishReason.STOP)])

    runtime = _runtime(generate_fn=_generate)
    runtime.batch_host = _BatchHost()

    events = asyncio.run(
        _execute_generation(
            runtime,
            "prompt",
            GenerateOptions(),
            input_embeddings=object(),
        )
    )

    assert called == {"batch": 0, "generate": 1}
    assert [event.text for event in events] == ["fallback"]


def test_stream_generation_uses_batch_host_stream_execute() -> None:
    called = {"stream": 0, "execute": 0}

    class _BatchHost:
        async def execute(self, prompt: str, options: GenerateOptions) -> list[TokenEvent]:
            del prompt, options
            called["execute"] += 1
            return [TokenEvent(token_id=9, text="buffered", finish_reason=FinishReason.STOP)]

        async def stream_execute(self, prompt: str, options: GenerateOptions):
            del prompt, options
            called["stream"] += 1
            yield TokenEvent(token_id=1, text="a")
            yield TokenEvent(token_id=2, text="b", finish_reason=FinishReason.STOP)

    runtime = _runtime()
    runtime.batch_host = _BatchHost()

    async def _run() -> None:
        seen = []
        async for event in _stream_generation(runtime, "prompt", GenerateOptions()):
            seen.append(event.text)
        assert seen == ["a", "b"]

    asyncio.run(_run())
    assert called == {"stream": 1, "execute": 0}
