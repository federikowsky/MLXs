"""Adaptive KV resident execution-view helpers."""

from __future__ import annotations

from dataclasses import dataclass

from mlxs.adaptive_kv.block_types import ResidentProfile
from mlxs.adaptive_kv.resident import (
    ResidentEncodedState,
    ResidentExecutionMode,
    ResidentStorageKind,
)


@dataclass(frozen=True, slots=True)
class ResidentAttentionSegment:
    """One ordered resident execution segment."""

    profile: ResidentProfile
    token_count: int
    logical_span: tuple[int, int]
    visible_span: tuple[int, int]
    block_slices: tuple[tuple[int, int, int], ...]
    resident_slice: tuple[int, int]
    q_keys: ResidentEncodedState
    q_values: ResidentEncodedState
    group_size: int
    bits: int
    storage_kind: ResidentStorageKind
    execution_mode: ResidentExecutionMode


@dataclass(frozen=True, slots=True)
class ResidentStateView:
    """Ordered resident execution view consumed by adaptive attention."""

    total_tokens: int
    segments: tuple[ResidentAttentionSegment, ...]

    @property
    def has_resident_segments(self) -> bool:
        return bool(self.segments)
