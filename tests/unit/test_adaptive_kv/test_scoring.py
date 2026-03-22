from __future__ import annotations

from mlxs.adaptive_kv.block_types import BlockRecord, BlockTier, PinState
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.scoring import AdaptiveScoreEngine


def _block(structural_prior: float = 0.2) -> BlockRecord:
    return BlockRecord(
        block_id=0,
        start_token=0,
        end_token=4,
        source_start=0,
        source_end=4,
        segment_id=0,
        pin_state=PinState.NORMAL,
        tier=BlockTier.FULL,
        created_step=0,
        structural_prior=structural_prior,
    )


def test_score_updates_raise_hotness_and_persistence() -> None:
    engine = AdaptiveScoreEngine(AdaptiveKVConfig(enabled=True))
    block = _block()

    updated = engine.update(block, usage=1.0, step=1)

    assert updated.score.hotness > 0.0
    assert updated.score.persistence > 0.0
    assert 0.0 <= updated.score.composite <= 1.0


def test_age_penalty_increases_without_usage() -> None:
    engine = AdaptiveScoreEngine(AdaptiveKVConfig(enabled=True, age_lambda=0.25))
    block = _block()

    cold = engine.update(block, usage=0.0, step=1)
    colder = engine.update(cold, usage=0.0, step=2)

    assert colder.age_windows == 2
    assert colder.score.age_penalty > cold.score.age_penalty


def test_structural_prior_keeps_score_bounded() -> None:
    engine = AdaptiveScoreEngine(
        AdaptiveKVConfig(
            enabled=True,
            w_hot=0.0,
            w_persist=0.0,
            w_struct=1.0,
            w_age=0.0,
        )
    )
    block = _block(structural_prior=1.0)

    updated = engine.update(block, usage=0.0, step=1)

    assert updated.score.composite == 1.0

