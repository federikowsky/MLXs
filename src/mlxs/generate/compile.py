"""Compiled decode and warmup utilities (§6.8, AC12).

Wraps the model forward pass with mx.compile for decode-time optimization.
Provides warmup logic to trigger JIT compilation before serving.

All optimizations are fallback-safe: when disabled, behavior is identical
to the baseline (AC12).
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def make_compiled_step(
    model: nn.Module,
) -> callable:
    """Create a compiled single-token decode step.

    Compiles only the model forward pass (not cache update or sampling)
    to avoid recompilation from shape changes (Plan §A4).

    Args:
        model: The model to compile.

    Returns:
        A compiled function with signature (input_ids, cache) -> logits.
    """

    @mx.compile
    def _compiled_forward(input_ids: mx.array, cache: list) -> mx.array:
        return model(input_ids, cache=cache)

    return _compiled_forward


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
