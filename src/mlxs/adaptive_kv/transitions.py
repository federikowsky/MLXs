"""Adaptive resident-profile transition policy."""

from __future__ import annotations

from mlxs.adaptive_kv.block_types import BlockRecord, PinState, PressureState, ResidentProfile
from mlxs.adaptive_kv.config import AdaptiveKVConfig


class AdaptiveTransitionEngine:
    """Decides restoration and degradation between resident profiles."""

    def __init__(self, config: AdaptiveKVConfig) -> None:
        self._config = config

    def should_restore(
        self,
        block: BlockRecord,
        *,
        pressure: PressureState,
        step: int,
    ) -> bool:
        if block.profile is not ResidentProfile.TQ_AGGR:
            return False
        if pressure is not PressureState.NORMAL:
            return False
        if block.windows_in_profile < self._config.min_dwell_aggr:
            return False
        if block.last_degrade_step >= 0 and (
            step - block.last_degrade_step
        ) < self._config.restore_cooldown:
            return False
        return block.score.composite >= self._config.t_tq_safe_restore

    def should_degrade(
        self,
        block: BlockRecord,
        *,
        pressure: PressureState,
        recent_tail: set[int],
        step: int,
    ) -> bool:
        if block.profile is not ResidentProfile.TQ_SAFE:
            return False
        if block.pin_state is PinState.HARD:
            return False
        if block.block_id in recent_tail:
            return False
        if pressure is PressureState.HARD:
            return True
        if block.windows_in_profile < self._config.min_dwell_safe:
            return False
        if block.last_restore_step >= 0 and (
            step - block.last_restore_step
        ) < self._config.degrade_cooldown:
            return False
        return block.score.composite <= self._config.t_tq_safe_degrade
