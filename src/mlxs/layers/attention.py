"""Scaled dot-product attention — standard and quantized KV cache paths.

Used by all transformer model architectures. No dependency on cache or models.
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
from mlx.utils import tree_map

from mlxs.adaptive_kv.resident import ResidentExecutionMode


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


def _slice_attention_mask(
    mask: mx.array | str | None,
    *,
    resident_total_tokens: int,
    resident_slice: tuple[int, int],
    query_tokens: int,
) -> mx.array | None:
    if mask is None:
        return None
    start, end = resident_slice
    if isinstance(mask, str):
        q_indices = mx.arange(resident_total_tokens - query_tokens, resident_total_tokens)
        k_indices = mx.arange(start, end)
        return q_indices[:, None] >= k_indices[None]
    return mask[..., start:end]


def _slice_scores(
    queries: mx.array,
    slice_ref: Any,
    *,
    scale: float,
) -> mx.array:
    if slice_ref.execution_mode is ResidentExecutionMode.FAMILY_SPECIFIC:
        raise ValueError(
            "Family-specific adaptive attention execution must be resolved by the "
            "runtime before reaching the generic attention path"
        )
    return _full_segment_scores(queries, slice_ref.keys_view, scale=scale)


def _slice_output(weights: mx.array, slice_ref: Any) -> mx.array:
    if slice_ref.execution_mode is ResidentExecutionMode.FAMILY_SPECIFIC:
        raise ValueError(
            "Family-specific adaptive attention execution must be resolved by the "
            "runtime before reaching the generic attention path"
        )
    return _full_segment_output(weights, slice_ref.values_view)


def _adaptive_segmented_reference_attention(
    queries: mx.array,
    resident_state: Any,
    *,
    scale: float,
    mask: mx.array | str | None,
    sample_usage: bool,
    cache: Any | None = None,
) -> mx.array | tuple[mx.array, mx.array]:
    if not resident_state.slices:
        raise ValueError("Adaptive attention requires at least one resident slice")

    trace = getattr(cache, "record_perf_ns", None)
    sync_enabled = bool(getattr(cache, "perf_sync_enabled", lambda: False)())

    pass1_started_ns = time.perf_counter_ns()
    global_max: mx.array | None = None
    exp_sum: mx.array | None = None
    query_tokens = queries.shape[-2]
    for slice_ref in resident_state.slices:
        local_scores = _slice_scores(queries, slice_ref, scale=scale)
        local_mask = _slice_attention_mask(
            mask,
            resident_total_tokens=resident_state.total_tokens,
            resident_slice=slice_ref.resident_slice,
            query_tokens=query_tokens,
        )
        local_scores, _ = _apply_attention_mask(local_scores, local_mask)
        local_max = local_scores.max(axis=-1)
        if global_max is None or exp_sum is None:
            global_max = local_max
            exp_sum = mx.exp(local_scores - global_max[..., None]).sum(axis=-1)
            continue
        next_max = mx.maximum(global_max, local_max)
        exp_sum = (
            exp_sum * mx.exp(global_max - next_max)
            + mx.exp(local_scores - next_max[..., None]).sum(axis=-1)
        )
        global_max = next_max

    if global_max is None or exp_sum is None:
        raise ValueError("Adaptive attention requires at least one resident slice score pass")
    if sync_enabled:
        mx.eval(global_max, exp_sum)
    if trace is not None:
        trace("attention.pass1_ns", time.perf_counter_ns() - pass1_started_ns)

    pass2_started_ns = time.perf_counter_ns()
    out: mx.array | None = None
    usage_parts: list[mx.array] | None = [] if sample_usage else None
    for slice_ref in resident_state.slices:
        local_scores = _slice_scores(queries, slice_ref, scale=scale)
        local_mask = _slice_attention_mask(
            mask,
            resident_total_tokens=resident_state.total_tokens,
            resident_slice=slice_ref.resident_slice,
            query_tokens=query_tokens,
        )
        local_scores, _ = _apply_attention_mask(local_scores, local_mask)
        local_weights = mx.exp(local_scores - global_max[..., None]) / exp_sum[..., None]
        if usage_parts is not None:
            usage_parts.append(_usage_by_token(local_weights))
        local_out = _slice_output(local_weights, slice_ref)
        out = local_out if out is None else out + local_out

    if out is None:
        raise ValueError("Adaptive attention requires at least one resident slice output")
    if usage_parts is None:
        if sync_enabled:
            mx.eval(out)
        if trace is not None:
            trace("attention.pass2_ns", time.perf_counter_ns() - pass2_started_ns)
        return out
    usage_by_token = (
        usage_parts[0]
        if len(usage_parts) == 1
        else mx.concatenate(usage_parts, axis=0)
    )
    if sync_enabled:
        mx.eval(out, usage_by_token)
    if trace is not None:
        trace("attention.pass2_ns", time.perf_counter_ns() - pass2_started_ns)
    return out, usage_by_token


def adaptive_scaled_dot_product_attention(
    queries: mx.array,
    resident_state: Any,
    scale: float,
    mask: mx.array | str | None,
    *,
    sample_usage: bool,
    cache: Any | None = None,
) -> mx.array | tuple[mx.array, mx.array]:
    trace = getattr(cache, "record_perf_ns", None)
    sync_enabled = bool(getattr(cache, "perf_sync_enabled", lambda: False)())
    total_started_ns = time.perf_counter_ns()
    if len(resident_state.slices) == 1 and not sample_usage:
        slice_ref = resident_state.slices[0]
        if slice_ref.execution_mode is ResidentExecutionMode.FAMILY_SPECIFIC:
            raise ValueError(
                "Family-specific adaptive attention execution must be resolved by the "
                "runtime before reaching the generic attention path"
            )
        if slice_ref.logical_span == slice_ref.visible_span:
            out = mx.fast.scaled_dot_product_attention(
                queries,
                slice_ref.keys_view,
                slice_ref.values_view,
                scale=scale,
                mask=mask,
                sinks=None,
            )
            if sync_enabled:
                mx.eval(out)
            if trace is not None:
                trace("attention.fast_path_ns", time.perf_counter_ns() - total_started_ns)
                trace("attention.total_ns", time.perf_counter_ns() - total_started_ns)
            return out
    out = _adaptive_segmented_reference_attention(
        queries,
        resident_state,
        scale=scale,
        mask=mask,
        sample_usage=sample_usage,
        cache=cache,
    )
    if trace is not None:
        trace("attention.total_ns", time.perf_counter_ns() - total_started_ns)
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
    if hasattr(cache, "resident_state_for_execution"):
        if sinks is not None:
            raise ValueError("Adaptive segmented SDPA does not support attention sinks.")
        out = adaptive_scaled_dot_product_attention(
            queries,
            keys,
            scale=scale,
            mask=mask,
            sample_usage=cache.should_sample_usage(),
            cache=cache,
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
