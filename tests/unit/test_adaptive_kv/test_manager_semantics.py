from __future__ import annotations

from dataclasses import replace

import mlx.core as mx

from mlxs.adaptive_kv.block_types import BlockTier, PressureState
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.manager import AdaptiveKVManager
from mlxs.observability.metrics import InMemoryMetrics


def _kv_from_tokens(tokens: list[int], *, head_dim: int = 32) -> tuple[mx.array, mx.array]:
    base = mx.array(tokens, dtype=mx.float32).reshape(1, 1, len(tokens), 1)
    keys = mx.broadcast_to(base + 1.0, (1, 1, len(tokens), head_dim))
    values = mx.broadcast_to(base + 2.0, (1, 1, len(tokens), head_dim))
    return keys, values


class _ReplayModel:
    def __call__(self, inputs: mx.array, *, cache=None, input_embeddings=None):  # type: ignore[no-untyped-def]
        del input_embeddings
        if cache is not None:
            keys, values = _kv_from_tokens([int(token.item()) for token in inputs.reshape(-1)])
            for layer_cache in cache:
                layer_cache.update_and_fetch(keys, values)
        return mx.zeros((inputs.shape[0], inputs.shape[1], 8), dtype=mx.float32)


def _make_manager(
    *,
    prompt_tokens: list[int],
    hard_budget_bytes: int | None = None,
    recent_tail_protect_blocks: int = 2,
    min_dwell_full: int = 1,
    demote_cooldown: int = 1,
) -> AdaptiveKVManager:
    config = AdaptiveKVConfig(
        enabled=True,
        block_size_tokens=2,
        update_window_steps=1,
        hard_budget_bytes=hard_budget_bytes,
        recent_tail_protect_blocks=recent_tail_protect_blocks,
        min_dwell_full=min_dwell_full,
        demote_cooldown=demote_cooldown,
        t_full_promote=0.95,
        t_full_demote=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=InMemoryMetrics())
    manager.bind_generation_context(model=_ReplayModel(), prefill_step_size=16)
    manager.initialize_prompt(prompt_tokens)
    keys, values = _kv_from_tokens(prompt_tokens)
    manager.caches()[0].update_and_fetch(keys, values)
    return manager


def test_replay_recovery_restores_evicted_block_as_resident() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4])
    block = manager.registry.get(1)

    manager._demote_block(block, reason="test_demote")
    manager._evict_block(manager.registry.get(1), reason="test_evict")

    assert manager.registry.get(1).tier is BlockTier.EVICTED
    assert manager.caches()[0].block_live_bytes(1) == 0

    manager.ensure_required_resident()

    recovered = manager.registry.get(1)
    assert recovered.tier is BlockTier.COMPRESSED
    assert manager.caches()[0].block_live_bytes(1) > 0


def test_manager_resident_bytes_use_live_tokens_not_slab_capacity() -> None:
    manager = _make_manager(prompt_tokens=[1, 2])
    keys, values = _kv_from_tokens([1, 2])

    assert manager.resident_bytes() == keys.nbytes + values.nbytes
    assert manager.caches()[0]._full_cache.state_size_bytes > manager.resident_bytes()


def test_hard_pressure_evicts_high_score_candidate_until_budget_work() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
    )
    middle = manager.registry.get(1)
    manager.registry.update(
        replace(
            middle,
            score=replace(middle.score, composite=1.0),
        )
    )

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._evict_if_needed(
        pressure=PressureState.HARD,
        protected=manager._protected_block_ids(),
    )

    assert manager.registry.get(1).tier is BlockTier.EVICTED


def test_hard_pressure_demotes_even_when_cooldown_would_normally_block() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
        min_dwell_full=5,
        demote_cooldown=4,
    )
    block = manager.registry.get(1)
    manager.registry.update(
        replace(
            block,
            windows_in_tier=0,
            last_promote_step=2,
        )
    )
    manager.decode_steps = 3

    manager._apply_transitions(
        pressure=PressureState.HARD,
        protected=manager._protected_block_ids(),
    )

    assert manager.registry.get(1).tier is BlockTier.COMPRESSED


def test_adjacent_compressed_blocks_coalesce_into_one_resident_segment() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        recent_tail_protect_blocks=0,
    )

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")

    segments = manager.caches()[0].resident_state_for_attention().segments
    compressed = [segment for segment in segments if segment.tier is BlockTier.COMPRESSED]

    assert len(compressed) == 1
    assert compressed[0].token_count == 4
    assert compressed[0].block_slices == ((1, 0, 2), (2, 2, 4))


def test_promoting_middle_block_splits_compressed_run() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        recent_tail_protect_blocks=0,
    )

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")
    manager._demote_block(manager.registry.get(3), reason="test_demote")
    manager._promote_block(manager.registry.get(2), reason="test_promote")

    segments = manager.caches()[0].resident_state_for_attention().segments
    compressed = [segment for segment in segments if segment.tier is BlockTier.COMPRESSED]

    assert len(compressed) == 2
    assert [segment.block_slices for segment in compressed] == [
        ((1, 0, 2),),
        ((3, 0, 2),),
    ]
