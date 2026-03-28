from __future__ import annotations

from mlxs.adaptive_kv.block_types import BlockRecord, PinState, ResidentProfile
from mlxs.adaptive_kv.ghost import AdaptiveGhostStore


def _block() -> BlockRecord:
    block = BlockRecord(
        block_id=7,
        start_token=0,
        end_token=4,
        source_start=0,
        source_end=4,
        segment_id=0,
        pin_state=PinState.NORMAL,
        profile=ResidentProfile.TQ_AGGR,
        created_step=0,
        structural_prior=0.0,
    )
    block.score.composite = 0.1
    return block


def test_ghost_record_creation() -> None:
    store = AdaptiveGhostStore()
    ghost = store.create(_block(), step=3)

    assert ghost.block_id == 7
    assert ghost.last_evicted_step == 3
    assert ghost.last_profile is ResidentProfile.TQ_AGGR
    assert store.has(7) is True


def test_repeated_eviction_increments_count() -> None:
    store = AdaptiveGhostStore()
    store.create(_block(), step=3)
    ghost = store.create(_block(), step=6)

    assert ghost.evict_count == 2


def test_mark_reactivated_sets_anti_thrash_flag() -> None:
    store = AdaptiveGhostStore()
    store.create(_block(), step=3)
    store.mark_reactivated(7)

    assert store.get(7) is not None
    assert store.get(7).recently_reactivated is True  # type: ignore[union-attr]
