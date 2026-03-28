from __future__ import annotations

from dataclasses import replace

from mlxs.adaptive_kv.block_types import BlockRecord, PinState, PressureState, ResidentProfile
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.transitions import AdaptiveTransitionEngine


def _block(profile: ResidentProfile) -> BlockRecord:
    block = BlockRecord(
        block_id=1,
        start_token=0,
        end_token=4,
        source_start=0,
        source_end=4,
        segment_id=0,
        pin_state=PinState.NORMAL,
        profile=profile,
        created_step=0,
        structural_prior=0.0,
    )
    block.score.composite = 0.8 if profile is ResidentProfile.TQ_AGGR else 0.2
    return replace(block, windows_in_profile=3)


def test_hysteresis_allows_restore_above_safe_threshold() -> None:
    engine = AdaptiveTransitionEngine(AdaptiveKVConfig(enabled=True))
    block = _block(ResidentProfile.TQ_AGGR)

    assert engine.should_restore(block, pressure=PressureState.NORMAL, step=3) is True


def test_recent_tail_blocks_do_not_degrade() -> None:
    engine = AdaptiveTransitionEngine(AdaptiveKVConfig(enabled=True))
    block = _block(ResidentProfile.TQ_SAFE)

    assert (
        engine.should_degrade(
            block,
            pressure=PressureState.NORMAL,
            recent_tail={block.block_id},
            step=3,
        )
        is False
    )


def test_hard_pinned_blocks_do_not_degrade() -> None:
    engine = AdaptiveTransitionEngine(AdaptiveKVConfig(enabled=True))
    block = replace(_block(ResidentProfile.TQ_SAFE), pin_state=PinState.HARD)

    assert (
        engine.should_degrade(
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
            min_dwell_safe=5,
            degrade_cooldown=4,
        )
    )
    block = replace(
        _block(ResidentProfile.TQ_SAFE),
        windows_in_profile=0,
        last_restore_step=2,
    )

    assert (
        engine.should_degrade(
            block,
            pressure=PressureState.HARD,
            recent_tail=set(),
            step=3,
        )
        is True
    )


def test_cooldown_blocks_immediate_restore() -> None:
    engine = AdaptiveTransitionEngine(
        AdaptiveKVConfig(enabled=True, restore_cooldown=2)
    )
    block = replace(_block(ResidentProfile.TQ_AGGR), last_degrade_step=2)

    assert engine.should_restore(block, pressure=PressureState.NORMAL, step=3) is False
