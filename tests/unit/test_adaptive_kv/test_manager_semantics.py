from __future__ import annotations

from dataclasses import replace
from typing import Any

import mlx.core as mx
import pytest

import mlxs.layers.attention as attention_mod
from mlxs.adaptive_kv.block_types import BlockTier, PressureState
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.manager import AdaptiveKVManager, AdaptiveLayerCache
from mlxs.adaptive_kv.metrics import (
    POST_RECOVERY_DECODE_FORWARDS_TOTAL,
    POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL,
    RECOVERY_MATERIALIZATION_EVENTS_TOTAL,
    RECOVERY_MATERIALIZATION_TIME_SECONDS_TOTAL,
    REPLAY_FORWARD_EVENTS_TOTAL,
    REPLAY_FORWARD_TIME_SECONDS_TOTAL,
)
from mlxs.adaptive_kv.storage import AdaptiveAttentionSegment, AdaptiveResidentState
from mlxs.layers.attention import (
    _adaptive_segmented_reference_attention,
    adaptive_scaled_dot_product_attention,
    scaled_dot_product_attention,
)
from mlxs.observability.metrics import InMemoryMetrics


def _kv_from_tokens(tokens: list[int], *, head_dim: int = 32) -> tuple[mx.array, mx.array]:
    base = mx.array(tokens, dtype=mx.float32).reshape(1, 1, len(tokens), 1)
    keys = mx.broadcast_to(base + 1.0, (1, 1, len(tokens), head_dim))
    values = mx.broadcast_to(base + 2.0, (1, 1, len(tokens), head_dim))
    return keys, values


def _q_from_tokens(tokens: list[int], *, head_dim: int = 32) -> mx.array:
    base = mx.array(tokens, dtype=mx.float32).reshape(1, 1, len(tokens), 1)
    return mx.broadcast_to(base, (1, 1, len(tokens), head_dim))


class _ReplayModel:
    def __init__(self) -> None:
        self.replayed_tokens = 0
        self.replay_chunks: list[int] = []

    def __call__(self, inputs: mx.array, *, cache=None, input_embeddings=None):  # type: ignore[no-untyped-def]
        del input_embeddings
        n_tokens = int(inputs.shape[1])
        self.replayed_tokens += n_tokens
        self.replay_chunks.append(n_tokens)
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


def _recover_block_in_hard_episode(manager: AdaptiveKVManager, block_id: int) -> None:
    manager._pressure_state = PressureState.HARD
    manager._update_hard_episode_state(PressureState.HARD)
    manager._demote_block(manager.registry.get(block_id), reason="test_demote")
    manager._evict_block(manager.registry.get(block_id), reason="test_evict")
    manager.ensure_required_resident()


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


def test_replay_only_rebuilds_prefix_needed_for_requested_blocks() -> None:
    replay_model = _ReplayModel()
    config = AdaptiveKVConfig(
        enabled=True,
        block_size_tokens=2,
        update_window_steps=1,
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
        t_full_promote=0.95,
        t_full_demote=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=InMemoryMetrics())
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    replay_model.replayed_tokens = 0
    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()

    assert manager.registry.get(1).tier is BlockTier.COMPRESSED
    assert replay_model.replayed_tokens == manager.registry.get(1).end_token


def test_recovery_records_replay_and_materialization_timing() -> None:
    replay_model = _ReplayModel()
    metrics = InMemoryMetrics()
    config = AdaptiveKVConfig(
        enabled=True,
        block_size_tokens=2,
        update_window_steps=1,
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
        t_full_promote=0.95,
        t_full_demote=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=metrics)
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()

    assert metrics.get_counter(REPLAY_FORWARD_EVENTS_TOTAL) == 1
    assert metrics.get_counter(REPLAY_FORWARD_TIME_SECONDS_TOTAL) > 0
    assert metrics.get_counter(RECOVERY_MATERIALIZATION_EVENTS_TOTAL) == 1
    assert metrics.get_counter(RECOVERY_MATERIALIZATION_TIME_SECONDS_TOTAL) > 0


def test_post_recovery_decode_timing_only_starts_after_recovery_wave() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4])
    metrics = manager.metrics

    manager.record_decode_forward_time(0.25)
    assert metrics.get_counter(POST_RECOVERY_DECODE_FORWARDS_TOTAL) == 0
    assert metrics.get_counter(POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL) == 0

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()
    manager.record_decode_forward_time(0.5)

    assert metrics.get_counter(POST_RECOVERY_DECODE_FORWARDS_TOTAL) == 1
    assert metrics.get_counter(POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL) == 0.5


def test_scratch_replay_extends_incrementally_across_recovery_waves() -> None:
    replay_model = _ReplayModel()
    config = AdaptiveKVConfig(
        enabled=True,
        block_size_tokens=2,
        update_window_steps=1,
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
        t_full_promote=0.95,
        t_full_demote=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=InMemoryMetrics())
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()
    assert replay_model.replayed_tokens == 4
    assert manager._scratch_replayed_tokens == 4
    assert manager._scratch_replay_materialized is False

    replay_model.replayed_tokens = 0
    replay_model.replay_chunks = []
    manager._demote_block(manager.registry.get(2), reason="test_demote")
    manager._evict_block(manager.registry.get(2), reason="test_evict")
    manager.ensure_required_resident()

    assert manager.registry.get(2).tier is BlockTier.COMPRESSED
    assert replay_model.replayed_tokens == 2
    assert replay_model.replay_chunks == [2]
    assert manager._scratch_replayed_tokens == 6
    assert manager._scratch_replay_materialized is False


def test_scratch_replay_reuses_cached_prefix_without_new_forward_work() -> None:
    replay_model = _ReplayModel()
    config = AdaptiveKVConfig(
        enabled=True,
        block_size_tokens=2,
        update_window_steps=1,
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
        t_full_promote=0.95,
        t_full_demote=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=InMemoryMetrics())
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    manager._demote_block(manager.registry.get(2), reason="test_demote")
    manager._evict_block(manager.registry.get(2), reason="test_evict")
    manager.ensure_required_resident()
    assert replay_model.replayed_tokens == 6
    assert manager._scratch_replayed_tokens == 6
    assert manager._scratch_replay_materialized is False

    replay_model.replayed_tokens = 0
    replay_model.replay_chunks = []
    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()

    assert manager.registry.get(1).tier is BlockTier.COMPRESSED
    assert replay_model.replayed_tokens == 0
    assert replay_model.replay_chunks == []
    assert manager._scratch_replayed_tokens == 6
    assert manager._scratch_replay_materialized is False


def test_adjacent_recovery_groups_use_run_level_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_model = _ReplayModel()
    config = AdaptiveKVConfig(
        enabled=True,
        block_size_tokens=2,
        update_window_steps=1,
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
        t_full_promote=0.95,
        t_full_demote=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=InMemoryMetrics())
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager._evict_block(manager.registry.get(2), reason="test_evict")

    used_run_recovery = {"value": False}

    original_recover_blocks = AdaptiveLayerCache.recover_blocks

    def _recover_blocks(
        self: AdaptiveLayerCache,
        blocks: tuple[Any, ...],
        run_keys: mx.array,
        run_values: mx.array,
    ) -> None:
        if len(blocks) > 1:
            used_run_recovery["value"] = True
        original_recover_blocks(self, blocks, run_keys, run_values)

    def _recover_block_fail(
        self: AdaptiveLayerCache,
        block_id: int,
        block_keys: mx.array,
        block_values: mx.array,
    ) -> None:
        raise AssertionError(
            f"Per-block recovery should not be used for adjacent recovered blocks: {block_id}"
        )

    monkeypatch.setattr(AdaptiveLayerCache, "recover_blocks", _recover_blocks)
    monkeypatch.setattr(AdaptiveLayerCache, "recover_block", _recover_block_fail)

    manager.ensure_required_resident()

    assert used_run_recovery["value"] is True
    assert manager.registry.get(1).tier is BlockTier.COMPRESSED
    assert manager.registry.get(2).tier is BlockTier.COMPRESSED


def test_recovered_block_is_not_re_evicted_in_same_hard_episode() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
    )

    _recover_block_in_hard_episode(manager, 1)

    manager._evict_if_needed(
        pressure=PressureState.HARD,
        protected=manager._protected_block_ids(),
    )

    assert manager.registry.get(1).tier is BlockTier.COMPRESSED
    snap = manager.debug_snapshot()["hard_stabilization"]
    assert snap["recovery_hold_active"] is True
    assert snap["best_achievable_under_current_forward_semantics"] is True
    assert snap["reason"] == "required_history_recovered_under_hard_episode"
    assert 1 in snap["stabilized_block_ids"]


def test_recovery_hold_blocks_new_wave_eviction_of_other_compressed_history() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
    )
    manager._pressure_state = PressureState.HARD
    manager._update_hard_episode_state(PressureState.HARD)

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()

    manager._evict_if_needed(
        pressure=PressureState.HARD,
        protected=manager._protected_block_ids(),
    )

    assert manager.registry.get(1).tier is BlockTier.COMPRESSED
    assert manager.registry.get(2).tier is BlockTier.COMPRESSED
    snap = manager.debug_snapshot()["hard_stabilization"]
    assert snap["recovery_hold_active"] is True
    assert snap["best_achievable_under_current_forward_semantics"] is True
    assert snap["reason"] == "required_history_recovered_under_hard_episode"


def test_hard_stabilization_resets_on_hard_to_soft() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
    )

    _recover_block_in_hard_episode(manager, 1)
    manager._pressure_state = PressureState.SOFT
    manager._update_hard_episode_state(PressureState.SOFT)

    snap = manager.debug_snapshot()["hard_stabilization"]
    assert snap["episode_active"] is False
    assert snap["stabilized_block_ids"] == []
    assert snap["recovery_hold_active"] is False
    assert snap["best_achievable_under_current_forward_semantics"] is False
    assert snap["reason"] is None


def test_hard_stabilization_resets_on_hard_to_normal() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
    )

    _recover_block_in_hard_episode(manager, 1)
    manager._pressure_state = PressureState.NORMAL
    manager._update_hard_episode_state(PressureState.NORMAL)

    snap = manager.debug_snapshot()["hard_stabilization"]
    assert snap["episode_active"] is False
    assert snap["stabilized_block_ids"] == []
    assert snap["recovery_hold_active"] is False
    assert snap["best_achievable_under_current_forward_semantics"] is False
    assert snap["reason"] is None


def test_best_achievable_over_budget_state_is_explicit() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
    )

    _recover_block_in_hard_episode(manager, 1)
    manager._evict_if_needed(
        pressure=PressureState.HARD,
        protected=manager._protected_block_ids(),
    )

    snap = manager.debug_snapshot()["hard_stabilization"]
    assert snap["recovery_hold_active"] is True
    assert snap["best_achievable_under_current_forward_semantics"] is True
    assert snap["reason"] == "required_history_recovered_under_hard_episode"
    assert snap["over_budget_bytes"] > 0
    assert snap["blocking_block_ids"]


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


def test_usage_timing_accumulates_only_at_window_flush() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]
    manager._adaptive_usage_timing_acc = {}

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    resident_state = cache.resident_state_for_attention()
    usage_by_token = mx.array([0.1, 0.2, 0.3, 0.4], dtype=mx.float32)

    cache.record_usage_from_attention(resident_state, usage_by_token)

    assert manager._adaptive_usage_timing_acc == {}

    usage = manager.usage.snapshot_and_reset(timing_acc=manager._adaptive_usage_timing_acc)

    assert usage
    assert manager._adaptive_usage_timing_acc.get("eval_ns", 0) > 0
    assert manager._adaptive_usage_timing_acc.get("host_ns", 0) >= 0


def test_record_usage_preserves_block_attribution_in_resident_order() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4, 5, 6], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")

    resident_state = cache.resident_state_for_attention()
    usage_by_token = mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=mx.float32)

    cache.record_usage_from_attention(resident_state, usage_by_token)
    usage = manager.usage.snapshot_and_reset()

    assert usage == pytest.approx(
        {
            0: 3.0 / 11.0,
            1: 7.0 / 11.0,
            2: 1.0,
        }
    )


def test_single_token_mixed_tier_attention_matches_reference_and_usage() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")

    resident_state = cache.resident_state_for_attention()
    queries = _q_from_tokens([9])
    assembled_keys = cache.keys
    assembled_values = cache.values

    assert assembled_keys is not None
    assert assembled_values is not None
    assert len(resident_state.segments) == 3

    out, usage_by_token = adaptive_scaled_dot_product_attention(
        queries,
        resident_state,
        scale=1.0,
        mask=None,
        sample_usage=True,
    )
    ref_out = mx.fast.scaled_dot_product_attention(
        queries,
        assembled_keys,
        assembled_values,
        scale=1.0,
        mask=None,
        sinks=None,
    )
    ref_scores = queries @ assembled_keys.swapaxes(-1, -2)
    ref_weights = mx.softmax(ref_scores, axis=-1, precise=True)
    ref_usage_by_token = ref_weights.mean(axis=(0, 1, 2))

    assert mx.allclose(out, ref_out, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(
        usage_by_token,
        ref_usage_by_token,
        rtol=1e-5,
        atol=1e-5,
    ).item()


def test_single_token_mixed_tier_attention_without_usage_matches_reference() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")

    resident_state = cache.resident_state_for_attention()
    queries = _q_from_tokens([9])
    assembled_keys = cache.keys
    assembled_values = cache.values

    assert assembled_keys is not None
    assert assembled_values is not None

    out = adaptive_scaled_dot_product_attention(
        queries,
        resident_state,
        scale=1.0,
        mask=None,
        sample_usage=False,
    )
    ref_out = mx.fast.scaled_dot_product_attention(
        queries,
        assembled_keys,
        assembled_values,
        scale=1.0,
        mask=None,
        sinks=None,
    )

    assert mx.allclose(out, ref_out, rtol=1e-5, atol=1e-5).item()


def test_single_token_mixed_tier_uses_decode_specialized_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")

    resident_state = cache.resident_state_for_attention()
    queries = _q_from_tokens([9])
    called = {"specialized": 0, "reference": 0}
    original_specialized = attention_mod._adaptive_decode_mixed_tier_attention
    original_reference = attention_mod._adaptive_segmented_reference_attention

    def _wrapped_specialized(*args: Any, **kwargs: Any) -> Any:
        called["specialized"] += 1
        return original_specialized(*args, **kwargs)

    def _wrapped_reference(*args: Any, **kwargs: Any) -> Any:
        called["reference"] += 1
        return original_reference(*args, **kwargs)

    monkeypatch.setattr(
        attention_mod,
        "_adaptive_decode_mixed_tier_attention",
        _wrapped_specialized,
    )
    monkeypatch.setattr(
        attention_mod,
        "_adaptive_segmented_reference_attention",
        _wrapped_reference,
    )

    adaptive_scaled_dot_product_attention(
        queries,
        resident_state,
        scale=1.0,
        mask=None,
        sample_usage=True,
    )

    assert called == {"specialized": 1, "reference": 0}


def test_multi_full_segment_single_token_path_matches_reference_and_usage() -> None:
    full_keys_a, full_values_a = _kv_from_tokens([1, 2])
    full_keys_b, full_values_b = _kv_from_tokens([5, 6])
    full_keys = mx.concatenate([full_keys_a, full_keys_b], axis=2)
    full_values = mx.concatenate([full_values_a, full_values_b], axis=2)
    segments = (
        AdaptiveAttentionSegment(
            tier=BlockTier.FULL,
            token_count=2,
            block_slices=((0, 0, 2),),
            resident_slice=(0, 2),
            full_slice=(0, 2),
        ),
        AdaptiveAttentionSegment(
            tier=BlockTier.FULL,
            token_count=2,
            block_slices=((3, 0, 2),),
            resident_slice=(2, 4),
            full_slice=(2, 4),
        ),
    )
    resident_state = AdaptiveResidentState(
        total_tokens=4,
        segments=segments,
        full_segments=segments,
        compressed_segments=(),
        full_keys=full_keys,
        full_values=full_values,
    )
    queries = _q_from_tokens([9])

    out, usage_by_token = adaptive_scaled_dot_product_attention(
        queries,
        resident_state,
        scale=1.0,
        mask=None,
        sample_usage=True,
    )
    ref_out = mx.fast.scaled_dot_product_attention(
        queries,
        full_keys,
        full_values,
        scale=1.0,
        mask=None,
        sinks=None,
    )
    ref_scores = queries @ full_keys.swapaxes(-1, -2)
    ref_weights = mx.softmax(ref_scores, axis=-1, precise=True)
    ref_usage_by_token = ref_weights.mean(axis=(0, 1, 2))

    assert mx.allclose(out, ref_out, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(
        usage_by_token,
        ref_usage_by_token,
        rtol=1e-5,
        atol=1e-5,
    ).item()


def test_multi_token_mixed_tier_falls_back_to_reference_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager._demote_block(manager.registry.get(1), reason="test_demote")
    manager._demote_block(manager.registry.get(2), reason="test_demote")

    resident_state = cache.resident_state_for_attention()
    queries = _q_from_tokens([9, 10])
    called = {"specialized": 0, "reference": 0}
    original_specialized = attention_mod._adaptive_decode_mixed_tier_attention
    original_reference = attention_mod._adaptive_segmented_reference_attention

    def _wrapped_specialized(*args: Any, **kwargs: Any) -> Any:
        called["specialized"] += 1
        return original_specialized(*args, **kwargs)

    def _wrapped_reference(*args: Any, **kwargs: Any) -> Any:
        called["reference"] += 1
        return original_reference(*args, **kwargs)

    monkeypatch.setattr(
        attention_mod,
        "_adaptive_decode_mixed_tier_attention",
        _wrapped_specialized,
    )
    monkeypatch.setattr(
        attention_mod,
        "_adaptive_segmented_reference_attention",
        _wrapped_reference,
    )

    out, usage_by_token = adaptive_scaled_dot_product_attention(
        queries,
        resident_state,
        scale=1.0,
        mask=None,
        sample_usage=True,
    )
    ref_out, ref_usage_by_token = _adaptive_segmented_reference_attention(
        queries,
        resident_state,
        full_keys=resident_state.full_keys,
        full_values=resident_state.full_values,
        scale=1.0,
        mask=None,
        sample_usage=True,
    )

    assert called == {"specialized": 0, "reference": 1}
    assert mx.allclose(out, ref_out, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(
        usage_by_token,
        ref_usage_by_token,
        rtol=1e-5,
        atol=1e-5,
    ).item()


def test_sampled_all_full_forward_keeps_fused_output_and_records_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager.before_decode_forward(5)
    assert cache.should_sample_usage() is True

    queries = _q_from_tokens([5])
    keys, values = _kv_from_tokens([5])
    resident_state, _ = cache.update_and_fetch(keys, values)
    sentinel = mx.full((1, 1, 1, queries.shape[-1]), 7.0, dtype=mx.float32)
    fast_calls = 0

    def fake_sdpa(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        scale: float,
        mask: mx.array | str | None,
        sinks: mx.array | None,
    ) -> mx.array:
        del q, k, v, scale, mask, sinks
        nonlocal fast_calls
        fast_calls += 1
        return sentinel

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", fake_sdpa)

    out = scaled_dot_product_attention(
        queries,
        resident_state,
        resident_state,
        cache=cache,
        scale=1.0,
        mask=cache.make_mask(1),
    )
    usage = manager.usage.snapshot_and_reset()
    mx.eval(out)

    assert fast_calls == 1
    assert out.tolist() == sentinel.tolist()
    assert usage
    assert manager.registry.block_for_token(cache.offset - 1).block_id in usage
