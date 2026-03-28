"""Adaptive KV memory- and execution-substrate view helpers."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlxs.adaptive_kv.block_types import ResidentProfile
from mlxs.adaptive_kv.resident import ResidentExecutionMode


@dataclass(frozen=True, slots=True)
class ExecutionSliceRef:
    """One ordered execution-visible slice over a persistent resident slab."""

    slab_id: int
    profile: ResidentProfile
    token_count: int
    logical_span: tuple[int, int]
    visible_span: tuple[int, int]
    resident_slice: tuple[int, int]
    block_slices: tuple[tuple[int, int, int], ...]
    local_slice: tuple[int, int]
    keys_view: mx.array
    values_view: mx.array
    execution_mode: ResidentExecutionMode


@dataclass(frozen=True, slots=True)
class ResidentSliceView:
    """Ordered visible slice view emitted by the resident memory substrate."""

    total_tokens: int
    slices: tuple[ExecutionSliceRef, ...]
    topology_epoch: int
    tail_epoch: int
    n_execution_slabs: int
    slab_token_counts: tuple[int, ...]
    fabric_compactions_total: int
    execution_view_topology_rebuilds_total: int

    @property
    def has_visible_slices(self) -> bool:
        return bool(self.slices)


@dataclass(frozen=True, slots=True)
class ExecutionPack:
    """Persistent exact execution-native unit built above the slab substrate."""

    pack_id: int
    token_count: int
    logical_span: tuple[int, int]
    block_slices: tuple[tuple[int, int, int], ...]
    keys: mx.array
    values: mx.array
    execution_mode: ResidentExecutionMode


@dataclass(frozen=True, slots=True)
class ExecutionPackView:
    """Ordered execution-visible view over a persistent execution pack."""

    pack_id: int
    token_count: int
    logical_span: tuple[int, int]
    visible_span: tuple[int, int]
    resident_slice: tuple[int, int]
    block_slices: tuple[tuple[int, int, int], ...]
    keys_view: mx.array
    values_view: mx.array
    execution_mode: ResidentExecutionMode


@dataclass(frozen=True, slots=True)
class ResidentStateView:
    """Ordered resident execution-pack view consumed by adaptive attention."""

    total_tokens: int
    packs: tuple[ExecutionPackView, ...]
    topology_epoch: int
    tail_epoch: int
    n_execution_packs: int
    pack_token_counts: tuple[int, ...]
    n_execution_slabs: int
    slab_token_counts: tuple[int, ...]
    fabric_compactions_total: int
    execution_view_topology_rebuilds_total: int
    observer_flushes_total: int = 0

    @property
    def slices(self) -> tuple[ExecutionPackView, ...]:
        """Compatibility alias for older slice-centric call sites."""
        return self.packs

    @property
    def has_visible_slices(self) -> bool:
        return bool(self.packs)
