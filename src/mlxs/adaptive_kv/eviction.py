"""Adaptive COMPRESSED -> EVICTED policy."""

from __future__ import annotations

from mlxs.adaptive_kv.block_types import BlockRecord, BlockTier, PinState, PressureState
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.ghost import AdaptiveGhostStore


class AdaptiveEvictionEngine:
    """Ranks compressed blocks for budget-aware eviction."""

    def __init__(self, config: AdaptiveKVConfig, ghost_store: AdaptiveGhostStore) -> None:
        self._config = config
        self._ghost_store = ghost_store

    def select_candidates(
        self,
        blocks: list[BlockRecord],
        *,
        pressure: PressureState,
        recent_tail: set[int],
    ) -> list[BlockRecord]:
        if pressure is not PressureState.HARD:
            return []
        candidates = [
            block
            for block in blocks
            if block.tier is BlockTier.COMPRESSED
            and block.pin_state is PinState.NORMAL
            and block.block_id not in recent_tail
        ]
        candidates.sort(key=self._rank_key)
        return candidates

    def eligible(self, block: BlockRecord, *, pressure: PressureState) -> bool:
        if pressure is PressureState.HARD:
            return True
        return block.score.composite <= self._config.t_evict_candidate

    def _rank_key(self, block: BlockRecord) -> tuple[float, int, int]:
        ghost = self._ghost_store.get(block.block_id)
        anti_thrash = 1 if ghost is not None and ghost.recently_reactivated else 0
        return (
            anti_thrash,
            int(block.score.composite * 1000),
            -block.age_windows,
        )
