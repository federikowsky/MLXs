from __future__ import annotations

from dataclasses import replace

from mlxs.adaptive_kv.block_types import BlockRecord, BlockTier, PinState, PressureState
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.transitions import AdaptiveTransitionEngine


def _block(tier: BlockTier) -> BlockRecord:
    block = BlockRecord(
        block_id=1,
        start_token=0,
        end_token=4,
        source_start=0,
        source_end=4,
        segment_id=0,
        pin_state=PinState.NORMAL,
        tier=tier,
        created_step=0,
        structural_prior=0.0,
    )
    block.score.composite = 0.8 if tier is BlockTier.COMPRESSED else 0.2
    return replace(block, windows_in_tier=3)


def test_hysteresis_allows_promotion_above_promote_threshold() -> None:
    engine = AdaptiveTransitionEngine(AdaptiveKVConfig(enabled=True))
    block = _block(BlockTier.COMPRESSED)

    assert engine.should_promote(block, pressure=PressureState.NORMAL, step=3) is True


def test_recent_tail_blocks_do_not_demote() -> None:
    engine = AdaptiveTransitionEngine(AdaptiveKVConfig(enabled=True))
    block = _block(BlockTier.FULL)

    assert (
        engine.should_demote(
            block,
            pressure=PressureState.NORMAL,
            recent_tail={block.block_id},
            step=3,
        )
        is False
    )


def test_hard_pinned_blocks_do_not_demote() -> None:
    engine = AdaptiveTransitionEngine(AdaptiveKVConfig(enabled=True))
    block = replace(_block(BlockTier.FULL), pin_state=PinState.HARD)

    assert (
        engine.should_demote(
            block,
            pressure=PressureState.HARD,
            recent_tail=set(),
            step=3,
        )
        is False
    )


def test_hard_pressure_bypasses_dwell_and_cooldown_for_non_protected_block() -> None:
    engine = AdaptiveTransitionEngine(
        AdaptiveKVConfig(
            enabled=True,
            min_dwell_full=5,
            demote_cooldown=4,
        )
    )
    block = replace(
        _block(BlockTier.FULL),
        windows_in_tier=0,
        last_promote_step=2,
    )

    assert (
        engine.should_demote(
            block,
            pressure=PressureState.HARD,
            recent_tail=set(),
            step=3,
        )
        is True
    )


def test_cooldown_blocks_immediate_repromotion() -> None:
    engine = AdaptiveTransitionEngine(
        AdaptiveKVConfig(enabled=True, promote_cooldown=2)
    )
    block = replace(_block(BlockTier.COMPRESSED), last_demote_step=2)

    assert engine.should_promote(block, pressure=PressureState.NORMAL, step=3) is False
