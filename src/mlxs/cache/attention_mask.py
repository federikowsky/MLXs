"""Attention mask utilities — shared across cache types.

Compatible with mlx.fast.scaled_dot_product_attention mask conventions:
- None: no masking needed (single token decode)
- "causal": string hint for fast-path causal masking
- mx.array: explicit boolean mask
"""

from __future__ import annotations

import mlx.core as mx


def create_causal_mask(
    n: int,
    offset: int = 0,
    window_size: int | None = None,
) -> mx.array:
    """Create a causal attention mask of shape ``(N, N + offset)``.

    Args:
        n: Query sequence length.
        offset: Number of past tokens (from KV cache).
        window_size: Optional sliding window size.

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
    return mask


def create_attention_mask(
    n: int,
    offset: int = 0,
    *,
    return_array: bool = False,
    window_size: int | None = None,
) -> mx.array | str | None:
    """Create attention mask for scaled_dot_product_attention.

    Args:
        n: Query sequence length.
        offset: KV cache offset (past tokens).
        return_array: Force array output instead of string hint.
        window_size: Optional sliding window size.

    Returns:
        None for single-token decode (no mask needed),
        "causal" string for fast-path causal masking,
        or an explicit boolean mask array.
    """
    if n == 1:
        return None
    if return_array or (window_size is not None and n > window_size):
        return create_causal_mask(n, offset=offset, window_size=window_size)
    return "causal"
