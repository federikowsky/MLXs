from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest

from mlxs.adaptive_kv.block_types import PressureState, ResidentProfile
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.manager import AdaptiveKVManager
from mlxs.adaptive_kv.metrics import (
    POST_RECOVERY_DECODE_FORWARDS_TOTAL,
    POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL,
    RECOVERY_MATERIALIZATION_EVENTS_TOTAL,
    RECOVERY_MATERIALIZATION_TIME_SECONDS_TOTAL,
    REPLAY_FORWARD_EVENTS_TOTAL,
    REPLAY_FORWARD_TIME_SECONDS_TOTAL,
)
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
    min_dwell_safe: int = 1,
    degrade_cooldown: int = 1,
) -> AdaptiveKVManager:
    config = AdaptiveKVConfig(
        enabled=True,
        block_size_tokens=2,
        update_window_steps=1,
        hard_budget_bytes=hard_budget_bytes,
        recent_tail_protect_blocks=recent_tail_protect_blocks,
        min_dwell_safe=min_dwell_safe,
        degrade_cooldown=degrade_cooldown,
        t_tq_safe_restore=0.95,
        t_tq_safe_degrade=0.9,
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
    manager._degrade_block(manager.registry.get(block_id), reason="test_degrade")
    manager._evict_block(manager.registry.get(block_id), reason="test_evict")
    manager.ensure_required_resident()


def test_replay_recovery_restores_evicted_block_as_tq_aggr() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4])
    block = manager.registry.get(1)

    manager._degrade_block(block, reason="test_degrade")
    manager._evict_block(manager.registry.get(1), reason="test_evict")

    assert manager.registry.get(1).profile is ResidentProfile.EVICTED
    assert manager.caches()[0].block_live_bytes(1) == 0

    manager.ensure_required_resident()

    recovered = manager.registry.get(1)
    assert recovered.profile is ResidentProfile.TQ_AGGR
    assert manager.caches()[0].block_live_bytes(1) > 0


def test_replay_only_rebuilds_prefix_needed_for_requested_blocks() -> None:
    replay_model = _ReplayModel()
    config = AdaptiveKVConfig(
        enabled=True,
        block_size_tokens=2,
        update_window_steps=1,
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
        t_tq_safe_restore=0.95,
        t_tq_safe_degrade=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=InMemoryMetrics())
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    replay_model.replayed_tokens = 0
    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()

    assert manager.registry.get(1).profile is ResidentProfile.TQ_AGGR
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
        t_tq_safe_restore=0.95,
        t_tq_safe_degrade=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=metrics)
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
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

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
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
        t_tq_safe_restore=0.95,
        t_tq_safe_degrade=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=InMemoryMetrics())
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()
    assert replay_model.replayed_tokens == 4
    assert manager._scratch_replayed_tokens == 4
    assert manager._scratch_replay_materialized is False

    replay_model.replayed_tokens = 0
    replay_model.replay_chunks = []
    manager._degrade_block(manager.registry.get(2), reason="test_degrade")
    manager._evict_block(manager.registry.get(2), reason="test_evict")
    manager.ensure_required_resident()

    assert manager.registry.get(2).profile is ResidentProfile.TQ_AGGR
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
        t_tq_safe_restore=0.95,
        t_tq_safe_degrade=0.9,
        t_evict_candidate=0.99,
    )
    manager = AdaptiveKVManager(config, num_layers=1, metrics=InMemoryMetrics())
    manager.bind_generation_context(model=replay_model, prefill_step_size=16)
    manager.initialize_prompt([1, 2, 3, 4, 5, 6, 7, 8])
    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
    manager.caches()[0].update_and_fetch(keys, values)

    manager._degrade_block(manager.registry.get(2), reason="test_degrade")
    manager._evict_block(manager.registry.get(2), reason="test_evict")
    manager.ensure_required_resident()
    assert replay_model.replayed_tokens == 6
    assert manager._scratch_replayed_tokens == 6
    assert manager._scratch_replay_materialized is False

    replay_model.replayed_tokens = 0
    replay_model.replay_chunks = []
    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._evict_block(manager.registry.get(1), reason="test_evict")
    manager.ensure_required_resident()

    assert manager.registry.get(1).profile is ResidentProfile.TQ_AGGR
    assert replay_model.replayed_tokens == 0
    assert replay_model.replay_chunks == []
    assert manager._scratch_replayed_tokens == 6
    assert manager._scratch_replay_materialized is False


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

    assert manager.registry.get(1).profile is ResidentProfile.TQ_AGGR
    snap = manager.debug_snapshot()["hard_stabilization"]
    assert snap["recovery_hold_active"] is True
    assert snap["best_achievable_under_current_forward_semantics"] is True
    assert snap["reason"] == "required_history_recovered_under_hard_episode"
    assert 1 in snap["stabilized_block_ids"]


def test_hard_stabilization_resets_on_hard_exit() -> None:
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


def test_manager_resident_bytes_match_sum_of_live_handles() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4])

    total = sum(
        manager.caches()[0].block_live_bytes(block.block_id)
        for block in manager.registry.snapshot()
    )

    assert manager.resident_bytes() == total
    assert manager.resident_bytes() > 0


def test_hard_pressure_evicts_high_score_candidate_until_budget_work() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
    )
    middle = manager.registry.get(1)
    manager.registry.update(replace(middle, score=replace(middle.score, composite=1.0)))

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._evict_if_needed(
        pressure=PressureState.HARD,
        protected=manager._protected_block_ids(),
    )

    assert manager.registry.get(1).profile is ResidentProfile.EVICTED


def test_hard_pressure_degrades_even_when_cooldown_would_normally_block() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        hard_budget_bytes=1,
        recent_tail_protect_blocks=0,
        min_dwell_safe=5,
        degrade_cooldown=4,
    )
    block = manager.registry.get(1)
    manager.registry.update(
        replace(
            block,
            windows_in_profile=0,
            last_restore_step=2,
        )
    )
    manager.decode_steps = 3

    manager._apply_transitions(
        pressure=PressureState.HARD,
        protected=manager._protected_block_ids(),
    )

    assert manager.registry.get(1).profile is ResidentProfile.TQ_AGGR


def test_adjacent_tq_aggr_blocks_coalesce_into_one_visible_slice() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        recent_tail_protect_blocks=0,
    )

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._degrade_block(manager.registry.get(2), reason="test_degrade")

    slices = manager.caches()[0].resident_state_for_execution().slices
    aggr = [slice_ref for slice_ref in slices if slice_ref.profile is ResidentProfile.TQ_AGGR]

    assert len(aggr) == 1
    assert aggr[0].token_count == 4
    assert aggr[0].block_slices == ((1, 0, 2), (2, 2, 4))


def test_restoring_middle_block_splits_tq_aggr_visible_slices() -> None:
    manager = _make_manager(
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        recent_tail_protect_blocks=0,
    )

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._degrade_block(manager.registry.get(2), reason="test_degrade")
    manager._degrade_block(manager.registry.get(3), reason="test_degrade")
    manager._restore_block(manager.registry.get(2), reason="test_restore")

    slices = manager.caches()[0].resident_state_for_execution().slices
    aggr = [slice_ref for slice_ref in slices if slice_ref.profile is ResidentProfile.TQ_AGGR]

    assert len(aggr) == 2
    assert [segment.block_slices for segment in aggr] == [
        ((1, 0, 2),),
        ((3, 0, 2),),
    ]


def test_usage_timing_accumulates_only_at_window_flush() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]
    manager._adaptive_usage_timing_acc = {}

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    resident_state = cache.resident_state_for_execution()
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

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._degrade_block(manager.registry.get(2), reason="test_degrade")

    resident_state = cache.resident_state_for_execution()
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


def test_single_token_resident_attention_matches_reference_and_usage() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._degrade_block(manager.registry.get(2), reason="test_degrade")

    resident_state = cache.resident_state_for_execution()
    queries = _q_from_tokens([9])
    assembled_keys = cache.keys
    assembled_values = cache.values

    assert assembled_keys is not None
    assert assembled_values is not None
    assert len(resident_state.slices) == 3

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
    assert mx.allclose(usage_by_token, ref_usage_by_token, rtol=1e-5, atol=1e-5).item()


def test_multi_token_resident_attention_matches_reference_executor() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager._degrade_block(manager.registry.get(1), reason="test_degrade")
    manager._degrade_block(manager.registry.get(2), reason="test_degrade")

    resident_state = cache.resident_state_for_execution()
    queries = _q_from_tokens([9, 10])

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
        scale=1.0,
        mask=None,
        sample_usage=True,
    )

    assert mx.allclose(out, ref_out, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(usage_by_token, ref_usage_by_token, rtol=1e-5, atol=1e-5).item()


def test_sampled_all_safe_forward_records_usage() -> None:
    manager = _make_manager(prompt_tokens=[1, 2, 3, 4], recent_tail_protect_blocks=0)
    cache = manager.caches()[0]

    manager.before_decode_forward(5)
    assert cache.should_sample_usage() is True

    queries = _q_from_tokens([5])
    keys, values = _kv_from_tokens([5])
    resident_state, _ = cache.update_and_fetch(keys, values)

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

    assert out.shape == (1, 1, 1, queries.shape[-1])
    assert usage
    assert manager.registry.block_for_token(cache.offset - 1).block_id in usage
