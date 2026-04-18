"""Layer 4 bootstrap and sole composition root."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mlxs._types import StreamPolicy
from mlxs.advanced_engines.prompt_cache import PromptCacheOrchestrator
from mlxs.config.schema import AppConfig
from mlxs.product_surfaces.lifecycle import RuntimeLifecycle
from mlxs.product_surfaces.observability import configure_observability

if TYPE_CHECKING:
    from mlxs.product_surfaces.batched_serving import BatchServingHost

logger = logging.getLogger(__name__)


def _shutdown_grace_timeout(config: AppConfig) -> float:
    """Bound graceful shutdown wait for already-admitted requests."""
    request_timeout = getattr(getattr(config, "server", None), "request_timeout", 30.0)
    if request_timeout is None:
        return 30.0
    return min(30.0, max(0.5, float(request_timeout)))


def _queue_counts(request_queue: Any) -> tuple[int, int]:
    active = getattr(request_queue, "active_count", None)
    pending = getattr(request_queue, "pending_count", None)
    return int(active or 0), int(pending or 0)


async def _await_request_drain(
    request_queue: Any,
    *,
    timeout_s: float,
) -> dict[str, float | int | bool]:
    """Wait briefly for already-admitted requests to drain."""
    if request_queue is None:
        return {
            "drained": True,
            "waited_s": 0.0,
            "active_count": 0,
            "pending_count": 0,
        }
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = loop.time() + timeout_s
    while True:
        active, pending = _queue_counts(request_queue)
        if active == 0 and pending == 0:
            return {
                "drained": True,
                "waited_s": loop.time() - started,
                "active_count": active,
                "pending_count": pending,
            }
        if loop.time() >= deadline:
            return {
                "drained": False,
                "waited_s": loop.time() - started,
                "active_count": active,
                "pending_count": pending,
            }
        await asyncio.sleep(0.01)


@dataclass(slots=True)
class ProductRuntime:
    """Layer 4-owned runtime container used only at the product boundary.

    This is intentionally a Layer 4 composition container. It is not a lower-layer
    context object and must not be pushed into Layers 1-3 as an architectural input.
    """

    config: AppConfig
    model: Any
    tokenizer: Any
    prompt_cache: Any
    prompt_cache_orchestrator: PromptCacheOrchestrator
    metrics: Any
    generate_fn: Any
    batch_host: BatchServingHost | None
    request_queue: Any
    lifecycle: RuntimeLifecycle

    async def shutdown(self) -> None:
        self.lifecycle.mark_draining()
        active_count, pending_count = _queue_counts(self.request_queue)
        timeout_s = _shutdown_grace_timeout(self.config)
        logger.info(
            "Layer 4 shutdown starting "
            "(instance_id=%s, active=%d, pending=%d, grace_timeout_s=%.3f)",
            self.lifecycle.instance_id,
            active_count,
            pending_count,
            timeout_s,
        )
        drain = await _await_request_drain(
            self.request_queue,
            timeout_s=timeout_s,
        )
        if self.batch_host is not None:
            self.batch_host.shutdown()
        self.lifecycle.mark_stopped()
        logger.info(
            "Layer 4 shutdown complete in %.3fs "
            "(instance_id=%s, drained=%s, active=%d, pending=%d)",
            float(drain["waited_s"]),
            self.lifecycle.instance_id,
            bool(drain["drained"]),
            int(drain["active_count"]),
            int(drain["pending_count"]),
        )


def _serving_completion_batch_size(config: AppConfig) -> int:
    """Layer 4 serving policy for text batching.

    Product surfaces inherit the configured completion batch size once the
    Layer 1 grouped-decode path is remotely validated for real-model parity.
    """
    return config.batch.completion_batch_size


def _runtime_model_config(config: AppConfig):
    """Force eager model residency for Layer 4 runtime bootstrap.

    Product-surface health/readiness should only report ready once the model is
    actually resident. Lazy model loading remains a lower-level capability, but
    the Layer 4 composition root materializes the model before serving starts.
    """
    if config.model.preload or not config.model.lazy_load:
        return config.model
    logger.info("Overriding lazy model load for Layer 4 runtime bootstrap")
    return config.model.model_copy(update={"preload": True})


def _effective_eager_residency(config: AppConfig) -> bool:
    """Layer 4 bootstrap always forces eager model residency before ready."""
    del config
    return True


def _new_generation_stream() -> Any:
    import mlx.core as mx

    return mx.new_stream(mx.default_device())


def _make_generate_fn(
    config: AppConfig,
    generate_impl: Any,
) -> Any:
    """Build Layer 4 generate binding with optional dedicated stream ownership."""
    if config.generate.stream_policy != StreamPolicy.OVERLAP:
        return generate_impl

    generation_stream = _new_generation_stream()

    def _generate(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("execution_stream", generation_stream)
        return generate_impl(*args, **kwargs)

    return _generate


def create_runtime(config: AppConfig) -> ProductRuntime:
    """Construct the concrete Layer 4 runtime from product configuration."""
    import mlx.core as mx

    from mlxs.general_path import generate_single_request
    from mlxs.generate.compile import warmup
    from mlxs.load import load_model_and_tokenizer
    from mlxs.product_surfaces.batched_serving import BatchServingHost
    from mlxs.prompt_cache import PromptCache
    from mlxs.server.queue import RequestQueue

    lifecycle = RuntimeLifecycle(model_id=config.model.model_path or "default")
    metrics = configure_observability(config)

    if config.memory.wired_limit is not None:
        mx.set_wired_limit(config.memory.wired_limit)
    else:
        try:
            info = mx.device_info()
            recommended = info.get("max_recommended_working_set_size")
            if recommended:
                mx.set_wired_limit(recommended)
                logger.info("Set wired limit to %d bytes (device recommendation)", recommended)
        except Exception:
            pass

    logger.info("Loading model from %s", config.model.model_path)
    model_config = _runtime_model_config(config)
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_config.model_path,
            model_config,
        )
    except Exception as exc:
        lifecycle.mark_error(str(exc))
        raise

    if config.generate.warmup_after_load:
        warmup(model, model.make_cache, vocab_size=model.vocab_size)

    prompt_cache = PromptCache(config.prompt_cache)
    prompt_cache_orchestrator = PromptCacheOrchestrator(prompt_cache)
    request_queue = RequestQueue(
        max_size=config.server.max_queue_size,
        max_concurrent=config.server.max_concurrent_requests,
        timeout=config.server.request_timeout,
    )
    completion_batch_size = _serving_completion_batch_size(config)
    batch_host = (
        BatchServingHost(
            model=model,
            tokenizer=tokenizer,
            prefill_batch_size=config.batch.prefill_batch_size,
            completion_batch_size=completion_batch_size,
            prefill_step_size=config.batch.prefill_step_size,
            prompt_cache_orchestrator=prompt_cache_orchestrator,
            model_id=config.model.model_path or "default",
        )
        if completion_batch_size > 1
        else None
    )

    runtime = ProductRuntime(
        config=config,
        model=model,
        tokenizer=tokenizer,
        prompt_cache=prompt_cache,
        prompt_cache_orchestrator=prompt_cache_orchestrator,
        metrics=metrics,
        generate_fn=_make_generate_fn(config, generate_single_request),
        batch_host=batch_host,
        request_queue=request_queue,
        lifecycle=lifecycle,
    )
    lifecycle.mark_ready()
    startup_duration_s = (
        lifecycle.ready_at - lifecycle.started_at
        if lifecycle.ready_at is not None
        else 0.0
    )
    logger.info(
        "Layer 4 runtime ready in %.3fs "
        "(instance_id=%s, effective_eager_residency=%s, warmup_after_load=%s, metrics_enabled=%s)",
        startup_duration_s,
        lifecycle.instance_id,
        _effective_eager_residency(config),
        config.generate.warmup_after_load,
        config.observability.metrics_enabled,
    )
    return runtime
