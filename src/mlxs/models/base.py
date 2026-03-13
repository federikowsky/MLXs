"""Shared model utilities — base classes and attention helpers.

Provides BaseModelArgs for config deserialization, attention/SSM mask
helpers, and the scaled_dot_product_attention dispatcher used by all
architectures.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx.utils import tree_map

from mlxs.cache.attention_mask import create_attention_mask as _create_mask
from mlxs.cache.attention_mask import create_causal_mask  # noqa: F401 — re-export


@dataclass
class BaseModelArgs:
    """Base for per-architecture model args.

    Provides ``from_dict`` that filters unknown keys, allowing forward
    compatibility with newer config.json fields.
    """

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> BaseModelArgs:
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


def create_attention_mask(
    h: mx.array,
    cache: Any = None,
    window_size: int | None = None,
    return_array: bool = False,
) -> mx.array | str | None:
    """Create attention mask from hidden states and cache.

    Compatible with mlx_lm's mask conventions. Delegates to cache.make_mask
    if available, otherwise creates a causal mask.
    """
    n = h.shape[1]
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(n, return_array=return_array, window_size=window_size)
    if n == 1:
        return None
    if return_array or (window_size is not None and n > window_size):
        return _create_mask(n, return_array=True, window_size=window_size)
    return "causal"


def create_ssm_mask(
    h: mx.array,
    cache: Any = None,
) -> mx.array | None:
    """Create SSM mask for state-space models (Mamba, Jamba, etc.)."""
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: mx.array | None,
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    """SDPA for quantized KV cache (compatible with mlx_lm)."""
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(scores, *q_values, transpose=False, group_size=group_size, bits=bits)

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))
    return out


def scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: Any,
    scale: float,
    mask: mx.array | str | None,
    sinks: mx.array | None = None,
) -> mx.array:
    """Dispatch SDPA to quantized or standard path based on cache type."""
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )
