"""Adaptive FULL <-> COMPRESSED transition policy."""

from __future__ import annotations

from mlxs.adaptive_kv.block_types import BlockRecord, BlockTier, PinState, PressureState
from mlxs.adaptive_kv.config import AdaptiveKVConfig


class AdaptiveTransitionEngine:
    """Decides promotions and demotions between resident tiers."""

    def __init__(self, config: AdaptiveKVConfig) -> None:
        self._config = config

    def should_promote(
        self,
        block: BlockRecord,
        *,
        pressure: PressureState,
        step: int,
    ) -> bool:
        if block.tier is not BlockTier.COMPRESSED:
            return False
        if pressure is not PressureState.NORMAL:
            return False
        if block.windows_in_tier < self._config.min_dwell_compressed:
            return False
        if block.last_demote_step >= 0 and (
            step - block.last_demote_step
        ) < self._config.promote_cooldown:
            return False
        return block.score.composite >= self._config.t_full_promote

    def should_demote(
        self,
        block: BlockRecord,
        *,
        pressure: PressureState,
        recent_tail: set[int],
        step: int,
    ) -> bool:
        if block.tier is not BlockTier.FULL:
            return False
        if block.pin_state is PinState.HARD:
            return False
        if block.block_id in recent_tail:
            return False
        # Under hard pressure, any non-protected FULL block is demotable.
        # Dwell/cooldown remain ordinary-flow guards only.
        if pressure is PressureState.HARD:
            return True
        if block.windows_in_tier < self._config.min_dwell_full:
            return False
        if block.last_promote_step >= 0 and (
            step - block.last_promote_step
        ) < self._config.demote_cooldown:
            return False
        return block.score.composite <= self._config.t_full_demote
