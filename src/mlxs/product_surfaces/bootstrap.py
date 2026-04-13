"""Layer 4 bootstrap and sole composition root."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mlxs.advanced_engines.prompt_cache import PromptCacheOrchestrator
from mlxs.config.schema import AppConfig
from mlxs.product_surfaces.batched_serving import BatchServingHost
from mlxs.product_surfaces.lifecycle import RuntimeLifecycle
from mlxs.product_surfaces.observability import configure_observability

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProductRuntime:
    """Layer 4-owned runtime container used only at the product boundary.

    This is intentionally a Layer 4 composition container. It is not a lower-layer
    context object and must not be pushed into Layers 1–3 as an architectural input.
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

    def shutdown(self) -> None:
        if self.batch_host is not None:
            self.batch_host.shutdown()
        self.lifecycle.mark_stopped()


def _serving_completion_batch_size(config: AppConfig) -> int:
    """Current Layer 4 serving policy for text batching.

    Grouped decode remains disabled at the product surface until remote
    validation proves semantic parity for same-prompt concurrent requests.
    Layer 4 still hosts the scheduler-backed path so concurrency is serialized
    onto one runtime worker instead of fanning out unsafe per-request threads.
    """
    del config
    return 1


def create_runtime(config: AppConfig) -> ProductRuntime:
    """Construct the concrete Layer 4 runtime from product configuration."""
    import mlx.core as mx

    from mlxs.generate.compile import warmup
    from mlxs.general_path import generate_single_request
    from mlxs.load import load_model_and_tokenizer
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
    try:
        model, tokenizer = load_model_and_tokenizer(
            config.model.model_path,
            config.model,
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
        generate_fn=generate_single_request,
        batch_host=batch_host,
        request_queue=request_queue,
        lifecycle=lifecycle,
    )
    lifecycle.mark_ready()
    return runtime
