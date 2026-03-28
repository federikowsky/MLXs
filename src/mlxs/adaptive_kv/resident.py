"""TurboQuant-first resident backend contracts and concrete handle types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

from mlxs.adaptive_kv.block_types import ResidentProfile

ResidentEncodedState: TypeAlias = tuple[mx.array, ...]


class ResidentExecutionMode(StrEnum):
    DIRECT_COMPRESSED = "direct_compressed"
    DEQUANTIZE_ON_READ = "dequantize_on_read"
    FAMILY_SPECIFIC = "family_specific"


class ResidentStorageKind(StrEnum):
    BITPACKED_EXACT = "bitpacked_exact"
    AFFINE_QUANTIZED = "affine_quantized"


@dataclass(frozen=True, slots=True)
class ResidentProfileDescriptor:
    profile: ResidentProfile
    bits: int
    group_size: int
    execution_chunk_tokens: int
    storage_kind: ResidentStorageKind
    execution_mode: ResidentExecutionMode
    fidelity_rank: int
    memory_cost_class: int


@dataclass(frozen=True, slots=True)
class ResidentBlockHandle:
    block_id: int
    profile: ResidentProfile
    logical_span: tuple[int, int]
    q_keys: ResidentEncodedState
    q_values: ResidentEncodedState
    group_size: int
    bits: int
    execution_chunk_tokens: int
    storage_kind: ResidentStorageKind
    dtype: mx.Dtype
    execution_mode: ResidentExecutionMode

    @property
    def token_count(self) -> int:
        return self.logical_span[1] - self.logical_span[0]

    @property
    def live_bytes(self) -> int:
        return tree_reduce(lambda a, x: a + x.nbytes, (self.q_keys, self.q_values), 0)


@runtime_checkable
class ResidentBackend(Protocol):
    def descriptor(
        self,
        profile: ResidentProfile,
        *,
        k_head_dim: int,
        v_head_dim: int,
    ) -> ResidentProfileDescriptor: ...

    def create_handle(
        self,
        *,
        block_id: int,
        profile: ResidentProfile,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
    ) -> ResidentBlockHandle: ...

    def append_tokens(
        self,
        handle: ResidentBlockHandle,
        *,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
        dtype: mx.Dtype,
    ) -> ResidentBlockHandle: ...

    def convert_profile(
        self,
        handle: ResidentBlockHandle,
        *,
        profile: ResidentProfile,
        dtype: mx.Dtype,
    ) -> ResidentBlockHandle: ...

    def materialize(
        self,
        handle: ResidentBlockHandle,
    ) -> tuple[mx.array, mx.array]: ...


def _slice_quantized_state(
    state: ResidentEncodedState,
    start: int,
    end: int,
) -> ResidentEncodedState:
    return tree_map(lambda x: x[..., start:end, :], state)


def _view_dtype(dtype: mx.Dtype) -> mx.Dtype:
    if dtype.size == 1:
        return mx.uint8
    if dtype.size == 2:
        return mx.uint16
    if dtype.size == 4:
        return mx.uint32
    raise ValueError(f"TurboQuant exact packing does not support dtype size {dtype.size}")


def _encode_exact(array: mx.array) -> ResidentEncodedState:
    return (array.view(_view_dtype(array.dtype)),)


def _decode_exact(state: ResidentEncodedState, *, dtype: mx.Dtype) -> mx.array:
    return state[0].view(dtype)


class TurboQuantResidentBackend:
    """MLX-aligned TurboQuant resident backend used by all branch families.

    The branch backend is exact-first: resident payloads are stored as bitpacked
    views of the original tensors so execution materialization remains lossless.
    """

    def __init__(
        self,
        *,
        safe_bits: int,
        aggr_bits: int,
        safe_execution_mode: ResidentExecutionMode = ResidentExecutionMode.DEQUANTIZE_ON_READ,
        aggr_execution_mode: ResidentExecutionMode = ResidentExecutionMode.DEQUANTIZE_ON_READ,
    ) -> None:
        self._safe_bits = safe_bits
        self._aggr_bits = aggr_bits
        self._safe_execution_mode = safe_execution_mode
        self._aggr_execution_mode = aggr_execution_mode

    def descriptor(
        self,
        profile: ResidentProfile,
        *,
        k_head_dim: int,
        v_head_dim: int,
    ) -> ResidentProfileDescriptor:
        group_size = self._packed_group_size(
            k_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
        )
        if profile is ResidentProfile.TQ_SAFE:
            return ResidentProfileDescriptor(
                profile=profile,
                bits=self._safe_bits,
                group_size=group_size,
                execution_chunk_tokens=64,
                storage_kind=ResidentStorageKind.BITPACKED_EXACT,
                execution_mode=self._safe_execution_mode,
                fidelity_rank=2,
                memory_cost_class=2,
            )
        if profile is ResidentProfile.TQ_AGGR:
            return ResidentProfileDescriptor(
                profile=profile,
                bits=self._aggr_bits,
                group_size=group_size,
                execution_chunk_tokens=32,
                storage_kind=ResidentStorageKind.BITPACKED_EXACT,
                execution_mode=self._aggr_execution_mode,
                fidelity_rank=1,
                memory_cost_class=1,
            )
        raise ValueError("Resident backend descriptors are not defined for evicted blocks")

    def create_handle(
        self,
        *,
        block_id: int,
        profile: ResidentProfile,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
    ) -> ResidentBlockHandle:
        descriptor = self.descriptor(
            profile,
            k_head_dim=keys.shape[-1],
            v_head_dim=values.shape[-1],
        )
        return ResidentBlockHandle(
            block_id=block_id,
            profile=profile,
            logical_span=logical_span,
            q_keys=_encode_exact(keys),
            q_values=_encode_exact(values),
            group_size=descriptor.group_size,
            bits=descriptor.bits,
            execution_chunk_tokens=descriptor.execution_chunk_tokens,
            storage_kind=descriptor.storage_kind,
            dtype=keys.dtype,
            execution_mode=descriptor.execution_mode,
        )

    def append_tokens(
        self,
        handle: ResidentBlockHandle,
        *,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
        dtype: mx.Dtype,
    ) -> ResidentBlockHandle:
        existing_keys, existing_values = self.materialize(handle)
        merged_keys = mx.concatenate((existing_keys, keys), axis=2)
        merged_values = mx.concatenate((existing_values, values), axis=2)
        return self.create_handle(
            block_id=handle.block_id,
            profile=handle.profile,
            logical_span=logical_span,
            keys=merged_keys,
            values=merged_values,
        )

    def convert_profile(
        self,
        handle: ResidentBlockHandle,
        *,
        profile: ResidentProfile,
        dtype: mx.Dtype,
    ) -> ResidentBlockHandle:
        if handle.profile is profile:
            return handle
        del dtype
        descriptor = self.descriptor(
            profile,
            k_head_dim=handle.q_keys[0].shape[-1],
            v_head_dim=handle.q_values[0].shape[-1],
        )
        return ResidentBlockHandle(
            block_id=handle.block_id,
            profile=profile,
            logical_span=handle.logical_span,
            q_keys=handle.q_keys,
            q_values=handle.q_values,
            group_size=descriptor.group_size,
            bits=descriptor.bits,
            execution_chunk_tokens=descriptor.execution_chunk_tokens,
            storage_kind=descriptor.storage_kind,
            dtype=handle.dtype,
            execution_mode=descriptor.execution_mode,
        )

    def materialize(
        self,
        handle: ResidentBlockHandle,
    ) -> tuple[mx.array, mx.array]:
        if handle.storage_kind is ResidentStorageKind.BITPACKED_EXACT:
            return (
                _decode_exact(handle.q_keys, dtype=handle.dtype),
                _decode_exact(handle.q_values, dtype=handle.dtype),
            )
        keys = mx.dequantize(
            handle.q_keys[0],
            handle.q_keys[1],
            handle.q_keys[2],
            group_size=handle.group_size,
            bits=handle.bits,
            dtype=handle.dtype,
        )
        values = mx.dequantize(
            handle.q_values[0],
            handle.q_values[1],
            handle.q_values[2],
            group_size=handle.group_size,
            bits=handle.bits,
            dtype=handle.dtype,
        )
        return keys, values

    @staticmethod
    def slice_handle(
        handle: ResidentBlockHandle,
        *,
        local_start: int,
        local_end: int,
    ) -> ResidentBlockHandle:
        span_start = handle.logical_span[0] + local_start
        span_end = handle.logical_span[0] + local_end
        return ResidentBlockHandle(
            block_id=handle.block_id,
            profile=handle.profile,
            logical_span=(span_start, span_end),
            q_keys=_slice_quantized_state(handle.q_keys, local_start, local_end),
            q_values=_slice_quantized_state(handle.q_values, local_start, local_end),
            group_size=handle.group_size,
            bits=handle.bits,
            execution_chunk_tokens=handle.execution_chunk_tokens,
            storage_kind=handle.storage_kind,
            dtype=handle.dtype,
            execution_mode=handle.execution_mode,
        )

    @staticmethod
    def _packed_group_size(*, k_head_dim: int, v_head_dim: int) -> int:
        del k_head_dim, v_head_dim
        return 1


__all__ = [
    "ResidentBackend",
    "ResidentBlockHandle",
    "ResidentEncodedState",
    "ResidentExecutionMode",
    "ResidentProfileDescriptor",
    "ResidentStorageKind",
    "TurboQuantResidentBackend",
]
