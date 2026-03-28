"""Adaptive KV resident-profile enums and record types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ResidentProfile(StrEnum):
    TQ_SAFE = "tq_safe"
    TQ_AGGR = "tq_aggr"
    EVICTED = "evicted"


class PinState(StrEnum):
    NORMAL = "normal"
    SOFT = "soft"
    HARD = "hard"


class PressureState(StrEnum):
    NORMAL = "normal"
    SOFT = "soft"
    HARD = "hard"


@dataclass(slots=True)
class ScoreComponents:
    hotness: float = 0.0
    persistence: float = 0.0
    structural_prior: float = 0.0
    age_penalty: float = 0.0
    composite: float = 0.0
    last_usage: float = 0.0


@dataclass(slots=True)
class TransitionRecord:
    step: int
    from_profile: ResidentProfile
    to_profile: ResidentProfile
    reason: str


@dataclass(slots=True)
class BlockRecord:
    block_id: int
    start_token: int
    end_token: int
    source_start: int
    source_end: int
    segment_id: int
    pin_state: PinState
    profile: ResidentProfile
    created_step: int
    structural_prior: float
    age_windows: int = 0
    windows_in_profile: int = 0
    last_access_step: int | None = None
    last_transition_step: int = 0
    last_restore_step: int = -1
    last_degrade_step: int = -1
    score: ScoreComponents = field(default_factory=ScoreComponents)
    last_transition: TransitionRecord | None = None

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    @property
    def resident(self) -> bool:
        return self.profile is not ResidentProfile.EVICTED


@dataclass(slots=True)
class GhostRecord:
    block_id: int
    source_start: int
    source_end: int
    segment_id: int
    last_evicted_step: int
    evict_count: int = 1
    last_score: float = 0.0
    last_profile: ResidentProfile = ResidentProfile.EVICTED
    recently_reactivated: bool = False


@dataclass(frozen=True, slots=True)
class RecomputeRequest:
    block_ids: tuple[int, ...]
    source_spans: tuple[tuple[int, int], ...]
    reason: str
