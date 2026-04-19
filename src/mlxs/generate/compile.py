"""Legacy compile helpers for the compatibility generation surface.

Wraps the model forward pass with mx.compile for decode-time optimization.
Provides warmup logic to trigger JIT compilation before serving.

All optimizations are fallback-safe: when disabled, behavior is identical
to the baseline (AC12).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

COMPILED_DECODE_PROMPT_TOKEN_LIMIT = 512


def compile_decode_eligible(
    *,
    prompt_token_count: int,
) -> bool:
    """Return whether the current decode request is eligible for compiled decode."""
    return prompt_token_count <= COMPILED_DECODE_PROMPT_TOKEN_LIMIT


def make_compiled_step(
    model: nn.Module,
    cache: list,
) -> Callable[[mx.array], mx.array]:
    """Create a compiled single-token decode step.

    ``mx.compile`` only accepts array trees as *formal* arguments; ``KVCache``
    objects are not traceable, so the active cache list is **closed over** here
    and the compiled function takes only ``input_ids`` (§6.8, AC12).

    ``decode_loop`` should call the result as ``step(input_ids)`` — same as an
    uncompiled ``functools.partial(model, cache=cache)``.

    Args:
        model: The model to compile.
        cache: KV cache list for this generation (same instance as decode_loop).

    Returns:
        ``(input_ids) -> logits``; only ``input_ids`` participates in compilation.
    """

    @mx.compile
    def compiled_step(input_ids: mx.array) -> mx.array:
        return model(input_ids, cache=cache)

    return compiled_step


def warmup(
    model: nn.Module,
    cache_factory: callable,
    *,
    vocab_size: int = 32000,
) -> None:
    """Run a dummy forward pass to trigger JIT compilation (§6.8).

    This forces MLX to compile the model graph before actual inference,
    avoiding cold-start latency on the first real request.

    Args:
        model: The model to warm up.
        cache_factory: Callable that returns a fresh cache list (e.g. model.make_cache).
        vocab_size: Vocab size for dummy input.
    """
    logger.info("Running warmup forward pass...")
    cache = cache_factory()

    # Single dummy token — minimal computation
    dummy_input = mx.array([[0]])
    logits = model(dummy_input, cache=cache)
    mx.eval(logits)

    # Clean up warmup state
    del cache, logits, dummy_input
    mx.clear_cache()
    logger.info("Warmup complete")
