"""Compile warmup helpers for Decode Engine V5."""

from __future__ import annotations

import logging
from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlxs.protocols.cache import CacheProtocol

logger = logging.getLogger(__name__)


def warmup(
    model: nn.Module,
    cache_factory: Callable[[], list[CacheProtocol]],
    *,
    vocab_size: int = 32000,
) -> None:
    """Run a dummy forward pass to trigger JIT compilation."""
    del vocab_size
    logger.info("Running warmup forward pass...")
    cache = cache_factory()
    dummy_input = mx.array([[0]])
    logits = model(dummy_input, cache=cache)
    mx.eval(logits)
    del cache, logits, dummy_input
    mx.clear_cache()
    logger.info("Warmup complete")


__all__ = ["warmup"]
