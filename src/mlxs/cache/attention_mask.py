"""Attention mask utilities — shared across cache types.

Compatible with mlx.fast.scaled_dot_product_attention mask conventions:
- None: no masking needed (single token decode)
- "causal": string hint for fast-path causal masking
- mx.array: explicit boolean mask
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def create_causal_mask(
    n: int,
    offset: int = 0,
    window_size: int | None = None,
    right_padding: mx.array | None = None,
    left_padding: mx.array | None = None,
) -> mx.array:
    """Create a causal attention mask of shape ``(N, N + offset)``.

    Args:
        n: Query sequence length.
        offset: Number of past tokens (from KV cache).
        window_size: Optional sliding window size.
        right_padding: Per-batch right-padding lengths.
        left_padding: Per-batch left-padding lengths.

    Returns:
        Boolean mask where ``mask[i, j] = True`` means position i attends to j.
    """
    rinds = mx.arange(offset + n)
    linds = mx.arange(offset, offset + n) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + n) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def _mask_from_length(
    n: int,
    offset: int = 0,
    *,
    return_array: bool = False,
    window_size: int | None = None,
) -> mx.array | str | None:
    """Low-level: create attention mask from sequence length (used by cache internals)."""
    if n == 1:
        return None
    if return_array or (window_size is not None and n > window_size):
        return create_causal_mask(n, offset=offset, window_size=window_size)
    return "causal"


def create_attention_mask(
    h: mx.array,
    cache: Any = None,
    window_size: int | None = None,
    return_array: bool = False,
) -> mx.array | str | None:
    """Create attention mask from hidden states and cache (for use in model layers).

    Delegates to cache.make_mask if available, otherwise creates a causal mask.
    Compatible with mlx_lm mask conventions.
    """
    n = h.shape[1]
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(n, return_array=return_array, window_size=window_size)
    return _mask_from_length(n, return_array=return_array, window_size=window_size)


def create_ssm_mask(
    h: mx.array,
    cache: Any = None,
) -> mx.array | None:
    """Create SSM mask for state-space models (Mamba, Jamba, etc.)."""
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None
