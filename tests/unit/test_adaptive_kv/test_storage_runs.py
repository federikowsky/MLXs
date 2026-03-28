from __future__ import annotations

import mlx.core as mx

from mlxs.adaptive_kv.block_types import ResidentProfile
from mlxs.adaptive_kv.resident import ResidentExecutionMode, TurboQuantResidentBackend


def _state(length: int) -> tuple[mx.array, mx.array]:
    base = mx.arange(length * 32, dtype=mx.float32).reshape(1, 1, length, 32)
    return base, base + 100.0


def test_backend_create_append_and_dequantize_preserve_span_and_length() -> None:
    backend = TurboQuantResidentBackend(
        safe_bits=8,
        aggr_bits=4,
        safe_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
        aggr_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
    )
    keys_a, values_a = _state(2)
    keys_b, values_b = _state(2)

    handle = backend.create_handle(
        block_id=10,
        profile=ResidentProfile.TQ_SAFE,
        logical_span=(0, 2),
        keys=keys_a,
        values=values_a,
    )
    handle = backend.append_tokens(
        handle,
        logical_span=(0, 4),
        keys=keys_b,
        values=values_b,
        dtype=mx.float32,
    )
    keys, values = backend.materialize(handle)

    assert handle.logical_span == (0, 4)
    assert handle.token_count == 4
    assert keys.shape[2] == 4
    assert values.shape[2] == 4
    assert handle.live_bytes > 0


def test_backend_profile_conversion_updates_descriptor_fields() -> None:
    backend = TurboQuantResidentBackend(
        safe_bits=8,
        aggr_bits=4,
        safe_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
        aggr_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
    )
    keys, values = _state(2)
    handle = backend.create_handle(
        block_id=11,
        profile=ResidentProfile.TQ_SAFE,
        logical_span=(2, 4),
        keys=keys,
        values=values,
    )

    aggr = backend.convert_profile(
        handle,
        profile=ResidentProfile.TQ_AGGR,
        dtype=mx.float32,
    )

    assert aggr.profile is ResidentProfile.TQ_AGGR
    assert aggr.logical_span == (2, 4)
    assert aggr.bits == 4
    assert aggr.execution_mode is ResidentExecutionMode.DEQUANTIZE_ON_READ


def test_backend_slice_handle_preserves_local_offsets() -> None:
    backend = TurboQuantResidentBackend(
        safe_bits=8,
        aggr_bits=4,
        safe_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
        aggr_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
    )
    keys, values = _state(4)
    handle = backend.create_handle(
        block_id=12,
        profile=ResidentProfile.TQ_AGGR,
        logical_span=(8, 12),
        keys=keys,
        values=values,
    )

    sliced = backend.slice_handle(handle, local_start=1, local_end=3)
    d_keys, d_values = backend.materialize(sliced)

    assert sliced.logical_span == (9, 11)
    assert sliced.token_count == 2
    assert d_keys.shape[2] == 2
    assert d_values.shape[2] == 2
