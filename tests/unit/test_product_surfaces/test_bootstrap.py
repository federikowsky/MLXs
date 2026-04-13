from __future__ import annotations

import asyncio

from mlxs.config.schema import AppConfig
from mlxs.product_surfaces.bootstrap import (
    ProductRuntime,
    _await_request_drain,
    _queue_counts,
    _effective_eager_residency,
    _runtime_model_config,
    _serving_completion_batch_size,
    _shutdown_grace_timeout,
)
from mlxs.product_surfaces.lifecycle import RuntimeLifecycle


def test_serving_completion_batch_size_uses_batch_config() -> None:
    config = AppConfig()
    config = config.model_copy(
        update={
            "batch": config.batch.model_copy(
                update={
                    "completion_batch_size": 4,
                    "max_batch_size": 8,
                }
            )
        }
    )

    assert _serving_completion_batch_size(config) == 4


def test_runtime_model_config_forces_preload_when_lazy_enabled() -> None:
    config = AppConfig()

    runtime_model = _runtime_model_config(config)

    assert runtime_model.preload is True
    assert runtime_model.model_path == config.model.model_path


def test_runtime_model_config_preserves_explicit_eager_model_config() -> None:
    config = AppConfig().model_copy(
        update={
            "model": AppConfig().model.model_copy(update={"lazy_load": False, "preload": False})
        }
    )

    runtime_model = _runtime_model_config(config)

    assert runtime_model is config.model


def test_effective_eager_residency_is_true_for_layer4_runtime() -> None:
    assert _effective_eager_residency(AppConfig()) is True


def test_shutdown_grace_timeout_is_bounded_by_request_timeout() -> None:
    config = AppConfig()
    assert _shutdown_grace_timeout(config) == 30.0

    config = config.model_copy(
        update={"server": config.server.model_copy(update={"request_timeout": 2.0})}
    )
    assert _shutdown_grace_timeout(config) == 2.0

    config = config.model_copy(
        update={"server": config.server.model_copy(update={"request_timeout": 0.01})}
    )
    assert _shutdown_grace_timeout(config) == 0.5


def test_queue_counts_handles_missing_values() -> None:
    class _Queue:
        active_count = None
        pending_count = None

    assert _queue_counts(_Queue()) == (0, 0)


def test_await_request_drain_reports_timeout_state() -> None:
    class _Queue:
        active_count = 1
        pending_count = 1

    result = asyncio.run(_await_request_drain(_Queue(), timeout_s=0.01))

    assert result["drained"] is False
    assert result["active_count"] == 1
    assert result["pending_count"] == 1


def test_product_runtime_shutdown_waits_for_request_drain() -> None:
    class _Queue:
        active_count = 1
        pending_count = 1

    class _BatchHost:
        def __init__(self) -> None:
            self.shutdown_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

    async def _run() -> None:
        config = AppConfig().model_copy(
            update={"server": AppConfig().server.model_copy(update={"request_timeout": 1.0})}
        )
        queue = _Queue()
        batch_host = _BatchHost()
        runtime = ProductRuntime(
            config=config,
            model=object(),
            tokenizer=object(),
            prompt_cache=object(),
            prompt_cache_orchestrator=object(),  # type: ignore[arg-type]
            metrics=object(),
            generate_fn=object(),
            batch_host=batch_host,  # type: ignore[arg-type]
            request_queue=queue,
            lifecycle=RuntimeLifecycle(model_id="m"),
        )

        async def _release() -> None:
            await asyncio.sleep(0.02)
            queue.active_count = 0
            queue.pending_count = 0

        runtime.lifecycle.mark_ready()
        releaser = asyncio.create_task(_release())
        await runtime.shutdown()
        await releaser

        assert batch_host.shutdown_called is True
        assert runtime.lifecycle.stopped is True

    asyncio.run(_run())
