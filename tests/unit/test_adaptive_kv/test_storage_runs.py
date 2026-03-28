from __future__ import annotations

import mlx.core as mx

from mlxs.adaptive_kv.block_types import ResidentProfile
from mlxs.adaptive_kv.resident import ResidentExecutionMode, TurboQuantResidentBackend


def _state(length: int) -> tuple[mx.array, mx.array]:
    base = mx.arange(length * 32, dtype=mx.float32).reshape(1, 1, length, 32)
    return base, base + 100.0


def test_backend_append_extends_tail_fragment_in_place() -> None:
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
    first_fragment = handle.fragments[0]

    handle = backend.append_tokens(
        handle,
        logical_span=(0, 4),
        keys=keys_b,
        values=values_b,
        dtype=mx.float32,
    )
    keys, values = backend.materialize(handle)

    assert handle.logical_span == (0, 4)
    assert len(handle.fragments) == 1
    assert handle.fragments[0].slab_id == first_fragment.slab_id
    assert handle.fragments[0].local_start == first_fragment.local_start
    assert handle.fragments[0].local_end == 4
    assert keys.shape[2] == 4
    assert values.shape[2] == 4


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
    assert aggr.slab_capacity_tokens == 128
    assert aggr.execution_mode is ResidentExecutionMode.DEQUANTIZE_ON_READ


def test_backend_compaction_preserves_remaining_block_mappings() -> None:
    backend = TurboQuantResidentBackend(
        safe_bits=8,
        aggr_bits=4,
        safe_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
        aggr_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
    )
    lengths = [80, 80, 80]
    handles = []
    offset = 0
    for block_id, length in enumerate(lengths):
        keys, values = _state(length)
        handle = backend.create_handle(
            block_id=block_id,
            profile=ResidentProfile.TQ_SAFE,
            logical_span=(offset, offset + length),
            keys=keys,
            values=values,
        )
        handles.append(handle)
        offset += length

    slab_id = handles[0].fragments[0].slab_id
    assert handles[1].fragments[0].slab_id == slab_id
    assert handles[2].fragments[0].slab_id == slab_id

    backend.evict_handle(handles[1])

    assert handles[2].fragments[0].slab_id == slab_id
    assert handles[2].fragments[0].local_start == 80
    assert backend.compactions_total == 1


def test_backend_allocates_dedicated_oversize_slab_for_large_block() -> None:
    backend = TurboQuantResidentBackend(
        safe_bits=8,
        aggr_bits=4,
        safe_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
        aggr_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
    )
    keys, values = _state(300)
    handle = backend.create_handle(
        block_id=12,
        profile=ResidentProfile.TQ_SAFE,
        logical_span=(0, 300),
        keys=keys,
        values=values,
    )

    assert len(handle.fragments) == 1
    assert handle.slab_capacity_tokens == 256
    slab = backend.fabric.slabs_for_handles((handle,))[0]
    assert slab.capacity_tokens == 300
    assert slab.used_tokens == 300
