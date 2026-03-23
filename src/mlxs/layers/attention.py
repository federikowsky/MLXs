"""Scaled dot-product attention — standard and quantized KV cache paths.

Used by all transformer model architectures. No dependency on cache or models.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx.utils import tree_map

from mlxs.adaptive_kv.block_types import BlockTier


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


def _full_segment_scores(
    queries: mx.array,
    keys: mx.array,
    *,
    scale: float,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = keys.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    queries = queries * scale
    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        keys = mx.expand_dims(keys, axis=-3)
        scores = queries @ keys.swapaxes(-1, -2)
        return mx.reshape(scores, (B, n_q_heads, L, keys.shape[-2]))
    return queries @ keys.swapaxes(-1, -2)


def _full_segment_output(weights: mx.array, values: mx.array) -> mx.array:
    B, n_q_heads, L, _ = weights.shape
    n_kv_heads = values.shape[1]
    D = values.shape[-1]
    n_repeats = n_q_heads // n_kv_heads
    if n_repeats > 1:
        weights = mx.reshape(weights, (B, n_kv_heads, n_repeats, L, values.shape[2]))
        values = mx.expand_dims(values, axis=-3)
        out = weights @ values
        return mx.reshape(out, (B, n_q_heads, L, D))
    return weights @ values


def _quantized_segment_scores(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    *,
    scale: float,
    group_size: int,
    bits: int,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    queries = queries * scale
    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        scores = mx.quantized_matmul(
            queries,
            *q_keys,
            transpose=True,
            group_size=group_size,
            bits=bits,
        )
        return mx.reshape(scores, (B, n_q_heads, L, q_keys[0].shape[-2]))
    return mx.quantized_matmul(
        queries,
        *q_keys,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


def _quantized_segment_output(
    weights: mx.array,
    q_values: tuple[mx.array, mx.array, mx.array],
    *,
    group_size: int,
    bits: int,
) -> mx.array:
    B, n_q_heads, L, _ = weights.shape
    n_kv_heads = q_values[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    D = q_values[0].shape[-1] * (8 * mx.uint32.size // bits)
    if n_repeats > 1:
        weights = mx.reshape(weights, (B, n_kv_heads, n_repeats, L, weights.shape[-1]))
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)
        out = mx.quantized_matmul(
            weights,
            *q_values,
            transpose=False,
            group_size=group_size,
            bits=bits,
        )
        return mx.reshape(out, (B, n_q_heads, L, D))
    return mx.quantized_matmul(
        weights,
        *q_values,
        transpose=False,
        group_size=group_size,
        bits=bits,
    )


def _segment_scores(
    queries: mx.array,
    segment: Any,
    *,
    full_keys: mx.array | None,
    scale: float,
) -> mx.array:
    if segment.tier is BlockTier.FULL:
        if full_keys is None or segment.full_slice is None:
            raise ValueError("Adaptive FULL segment is missing contiguous resident tensors")
        start, end = segment.full_slice
        return _full_segment_scores(
            queries,
            full_keys[..., start:end, :],
            scale=scale,
        )
    if (
        segment.q_keys is None
        or segment.group_size is None
        or segment.bits is None
    ):
        raise ValueError("Adaptive COMPRESSED segment is missing quantized resident tensors")
    return _quantized_segment_scores(
        queries,
        segment.q_keys,
        scale=scale,
        group_size=segment.group_size,
        bits=segment.bits,
    )


def _segment_output(
    weights: mx.array,
    segment: Any,
    *,
    full_values: mx.array | None,
) -> mx.array:
    if segment.tier is BlockTier.FULL:
        if full_values is None or segment.full_slice is None:
            raise ValueError("Adaptive FULL segment is missing contiguous resident tensors")
        start, end = segment.full_slice
        return _full_segment_output(
            weights,
            full_values[..., start:end, :],
        )
    if (
        segment.q_values is None
        or segment.group_size is None
        or segment.bits is None
    ):
        raise ValueError("Adaptive COMPRESSED segment is missing quantized resident tensors")
    return _quantized_segment_output(
        weights,
        segment.q_values,
        group_size=segment.group_size,
        bits=segment.bits,
    )


def _usage_by_token(segment_weights: mx.array) -> mx.array:
    return segment_weights.mean(axis=(0, 1, 2))


def _apply_attention_mask(
    scores: mx.array,
    mask: mx.array | str | None,
) -> tuple[mx.array, mx.array | None]:
    if mask is None:
        return scores, None
    if isinstance(mask, str):
        qL, kL = scores.shape[-2:]
        q_indices = mx.arange(kL - qL, kL)
        k_indices = mx.arange(kL)
        mask = q_indices[:, None] >= k_indices[None]
    if mask.dtype == mx.bool_:
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
    else:
        scores = scores + mask
    return scores, mask


def _is_all_full_contiguous_resident_state(
    resident_state: Any,
    full_keys: mx.array | None,
    full_values: mx.array | None,
) -> bool:
    if full_keys is None or full_values is None or resident_state.has_compressed:
        return False
    if len(resident_state.segments) != 1:
        return False
    segment = resident_state.segments[0]
    return (
        segment.tier is BlockTier.FULL
        and segment.token_count == resident_state.total_tokens
        and segment.full_slice == (0, resident_state.total_tokens)
        and full_keys.shape[2] == resident_state.total_tokens
        and full_values.shape[2] == resident_state.total_tokens
    )


def adaptive_scaled_dot_product_attention(
    queries: mx.array,
    resident_state: Any,
    scale: float,
    mask: mx.array | str | None,
    *,
    sample_usage: bool,
) -> mx.array | tuple[mx.array, mx.array]:
    if resident_state.full_keys is None or resident_state.full_values is None:
        full_keys = None
        full_values = None
    else:
        full_keys = resident_state.full_keys
        full_values = resident_state.full_values

    if (
        _is_all_full_contiguous_resident_state(
            resident_state,
            full_keys,
            full_values,
        )
        and full_keys is not None
        and full_values is not None
    ):
        # Keep the generation result on fused SDPA when adaptive resident state is
        # semantically identical to the baseline contiguous FULL cache.
        out = mx.fast.scaled_dot_product_attention(
            queries,
            full_keys,
            full_values,
            scale=scale,
            mask=mask,
            sinks=None,
        )
        if not sample_usage:
            return out
        scores = _full_segment_scores(
            queries,
            full_keys,
            scale=scale,
        )
        scores, _ = _apply_attention_mask(scores, mask)
        weights = mx.softmax(scores, axis=-1, precise=True)
        return out, weights.mean(axis=(0, 1, 2))
    if (
        len(resident_state.segments) == 1
        and resident_state.segments[0].tier is BlockTier.COMPRESSED
        and not sample_usage
    ):
        segment = resident_state.segments[0]
        if (
            segment.q_keys is None
            or segment.q_values is None
            or segment.group_size is None
            or segment.bits is None
        ):
            raise ValueError("Adaptive COMPRESSED segment is missing quantized resident tensors")
        return quantized_scaled_dot_product_attention(
            queries,
            segment.q_keys,
            segment.q_values,
            scale=scale,
            mask=mask,
            group_size=segment.group_size,
            bits=segment.bits,
        )

    score_parts: list[mx.array] = []
    for segment in resident_state.segments:
        score_parts.append(
            _segment_scores(
                queries,
                segment,
                full_keys=full_keys,
                scale=scale,
            )
        )

    if not score_parts:
        raise ValueError("Adaptive attention requires at least one resident segment")

    scores = score_parts[0] if len(score_parts) == 1 else mx.concatenate(score_parts, axis=-1)
    scores, _ = _apply_attention_mask(scores, mask)
    weights = mx.softmax(scores, axis=-1, precise=True)

    out: mx.array | None = None
    usage_parts: list[mx.array] | None = [] if sample_usage else None
    cursor = 0
    for segment in resident_state.segments:
        segment_weights = weights[..., cursor : cursor + segment.token_count]
        cursor += segment.token_count
        if usage_parts is not None:
            usage_parts.append(_usage_by_token(segment_weights))
        segment_out = _segment_output(
            segment_weights,
            segment,
            full_values=full_values,
        )
        out = segment_out if out is None else out + segment_out
    if out is None:
        raise ValueError("Adaptive attention requires at least one resident segment output")
    if usage_parts is None:
        return out
    usage_by_token = (
        usage_parts[0]
        if len(usage_parts) == 1
        else mx.concatenate(usage_parts, axis=0)
    )
    return out, usage_by_token


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
    if hasattr(cache, "resident_state_for_attention"):
        if sinks is not None:
            raise ValueError("Adaptive segmented SDPA does not support attention sinks.")
        out = adaptive_scaled_dot_product_attention(
            queries,
            keys,
            scale=scale,
            mask=mask,
            sample_usage=cache.should_sample_usage(),
        )
        if isinstance(out, tuple):
            out_tensor, usage_by_token = out
            cache.record_usage_from_attention(keys, usage_by_token)
            return out_tensor
        return out
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        out = quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        out = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
    observer = getattr(cache, "observe_attention", None)
    if observer is not None:
        observer(queries=queries, keys=keys, scale=scale, mask=mask)
    return out
