"""Minimal token selection for Layer 1."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

TokenSelector = Callable[[mx.array], mx.array]


def greedy_select(logits: mx.array) -> mx.array:
    """Select the highest-logit token id."""
    return mx.argmax(logits, axis=-1)
