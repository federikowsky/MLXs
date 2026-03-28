"""Adaptive KV manager and cache wrapper."""

from __future__ import annotations

import os
import time
from enum import StrEnum
from typing import Any

from mlxs.adaptive_kv.adapters import default_generation_adapter
from mlxs.adaptive_kv.block_registry import AdaptiveBlockRegistry
from mlxs.adaptive_kv.block_types import (
    BlockRecord,
    PinState,
    PressureState,
    ResidentProfile,
    TransitionRecord,
)
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.eviction import AdaptiveEvictionEngine
from mlxs.adaptive_kv.exceptions import AdaptiveKVError, AdaptiveKVUnsupportedError
from mlxs.adaptive_kv.ghost import AdaptiveGhostStore
from mlxs.adaptive_kv.metrics import (
    DEGRADES_TOTAL,
    EVICTIONS_TOTAL,
    POLICY_TIME_SECONDS,
    POST_RECOVERY_DECODE_FORWARDS_TOTAL,
    POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL,
    PRESSURE_HARD_COUNT,
    PRESSURE_SOFT_COUNT,
    RECOMPUTATIONS_TOTAL,
    RECOVERY_MATERIALIZATION_EVENTS_TOTAL,
    RECOVERY_MATERIALIZATION_TIME_SECONDS_TOTAL,
    REPLAY_FORWARD_EVENTS_TOTAL,
    REPLAY_FORWARD_TIME_SECONDS_TOTAL,
    RESTORES_TOTAL,
    SCORE_UPDATES_TOTAL,
    block_debug_view,
    emit_population,
)
from mlxs.adaptive_kv.perf_trace import AdaptiveKVPerfTrace
from mlxs.adaptive_kv.recompute import AdaptiveRecomputeCoordinator
from mlxs.adaptive_kv.runtime import (
    AdaptiveKVLayerRuntime,
    AdaptiveKVReplayBackend,
    AdaptiveKVRuntimeAdapter,
    AdaptiveKVRuntimeSubstrate,
)
from mlxs.adaptive_kv.scoring import AdaptiveScoreEngine
from mlxs.adaptive_kv.transitions import AdaptiveTransitionEngine
from mlxs.adaptive_kv.usage import AdaptiveUsageCollector

_DEFAULT_LAYER_RUNTIME = default_generation_adapter().layer_runtime_type
if _DEFAULT_LAYER_RUNTIME is None:
    raise RuntimeError("Adaptive KV default adapter must expose a concrete layer runtime type")

AdaptiveLayerCache = _DEFAULT_LAYER_RUNTIME

_DORMANT_ENTRY_CALM_WINDOWS = 2
_DORMANT_INTERVAL_MULTIPLIER = 4
_SOFT_WAKE_FRACTION = 0.85
_POST_MUTATION_ACTIVE_WINDOWS = 2


class _ControlCadenceMode(StrEnum):
    DORMANT = "dormant"
    ACTIVE = "active"
    STRESSED = "stressed"


class AdaptiveKVManager:
    """Top-level coordinator for the TurboQuant-first Adaptive KV branch."""

    def __init__(
        self,
        config: AdaptiveKVConfig,
        *,
        num_layers: int,
        metrics: Any,
        runtime_adapter: AdaptiveKVRuntimeAdapter | None = None,
        runtime_substrate: AdaptiveKVRuntimeSubstrate | None = None,
        replay_backend: AdaptiveKVReplayBackend | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self.runtime_adapter = (
            runtime_adapter if runtime_adapter is not None else default_generation_adapter()
        )
        self.runtime_substrate = (
            runtime_substrate
            if runtime_substrate is not None
            else self.runtime_adapter.runtime_substrate
        )
        self.replay_backend = (
            replay_backend if replay_backend is not None else self.runtime_adapter.replay_backend
        )
        self.registry = AdaptiveBlockRegistry(config.block_size_tokens)
        self.usage = AdaptiveUsageCollector()
        self.scoring = AdaptiveScoreEngine(config)
        self.ghost_store = AdaptiveGhostStore()
        self.transitions = AdaptiveTransitionEngine(config)
        self.evictions = AdaptiveEvictionEngine(config, self.ghost_store)
        perf_enabled = os.environ.get("MLXS_ADAPTIVE_KV_PERF_ATTRIBUTION") == "1"
        self._perf_trace = AdaptiveKVPerfTrace(
            enabled=perf_enabled,
            sync_enabled=perf_enabled
            and os.environ.get("MLXS_ADAPTIVE_KV_PERF_ATTRIBUTION_SYNC", "1") == "1",
        )
        self.decode_steps = 0
        self.prompt_token_count = 0
        self.source_tokens: list[int] = []
        self._num_layers = num_layers
        self._layer_caches: list[AdaptiveKVLayerRuntime] = [
            self.runtime_substrate.make_layer_runtime(self, layer_index=i)
            for i in range(num_layers)
        ]
        self._resident_version = 0
        self._pressure_state = PressureState.NORMAL
        self._collect_usage_this_forward = False
        self._control_cadence_mode = _ControlCadenceMode.ACTIVE
        self._last_control_step = 0
        self._calm_control_windows = 0
        self._force_active_until_step = 0
        self._control_windows_total = 0
        self._control_skipped_decode_steps_total = 0
        self._usage_sample_steps_total = 0
        self._mutations_in_current_control_window = 0
        self._model: Any = None
        self._prefill_step_size = 2048
        self._pending_recompute_requests = 0
        self._adaptive_usage_timing_acc: dict[str, int] | None = None
        if self._perf_trace.enabled and self._adaptive_usage_timing_acc is None:
            self._adaptive_usage_timing_acc = {}
        self._hard_episode_active = False
        self._hard_episode_stabilized_blocks: set[int] = set()
        self._hard_episode_recovery_hold = False
        self._hard_best_achievable = False
        self._hard_best_achievable_reason: str | None = None
        self._hard_best_achievable_over_budget_bytes = 0
        self._hard_best_achievable_blocking_block_ids: tuple[int, ...] = ()
        self._scratch_replay_cache: list[Any] | None = None
        self._scratch_replayed_tokens = 0
        self._scratch_replay_materialized = True
        self._recovery_wave_seen = False
        self.recompute = AdaptiveRecomputeCoordinator(
            self.registry,
            on_request=self._increment_recompute_requests,
            on_recover=self._recover_request,
        )

    @property
    def resident_version(self) -> int:
        return self._resident_version

    def bind_generation_context(self, *, model: Any, prefill_step_size: int) -> None:
        self._model = model
        self._prefill_step_size = prefill_step_size
        self._reset_scratch_replay()
        self._recovery_wave_seen = False
        self._control_cadence_mode = _ControlCadenceMode.ACTIVE
        self._last_control_step = 0
        self._calm_control_windows = 0
        self._force_active_until_step = 0
        self._control_windows_total = 0
        self._control_skipped_decode_steps_total = 0
        self._usage_sample_steps_total = 0
        self._mutations_in_current_control_window = 0

    def bump_resident_version(self) -> None:
        self._resident_version += 1

    @property
    def perf_trace(self) -> AdaptiveKVPerfTrace:
        return self._perf_trace

    def record_perf_ns(self, name: str, elapsed_ns: int) -> None:
        self._perf_trace.record_ns(name, elapsed_ns)

    def increment_perf(self, name: str, delta: int = 1) -> None:
        self._perf_trace.increment(name, delta)

    def perf_sync_enabled(self) -> bool:
        return self._perf_trace.sync_enabled

    def caches(self) -> list[AdaptiveKVLayerRuntime]:
        return self._layer_caches

    def initialize_prompt(self, prompt_tokens: list[int]) -> None:
        self._reset_scratch_replay()
        self._recovery_wave_seen = False
        self.source_tokens = list(prompt_tokens)
        self.prompt_token_count = len(prompt_tokens)
        self.registry.initialize_prompt(len(prompt_tokens), step=0)
        self._emit_population()

    def ensure_block_coverage(self, total_tokens: int) -> None:
        self.registry.ensure_generated_tokens(total_tokens, step=self.decode_steps)

    def before_decode_forward(self, token_id: int) -> None:
        self.source_tokens.append(token_id)
        self._collect_usage_this_forward = self._should_sample_usage()
        if self._collect_usage_this_forward:
            self._usage_sample_steps_total += 1
            self.increment_perf("manager.usage_sample_steps_total")

    def ensure_required_resident(self) -> None:
        block_ids = self._required_evicted_block_ids()
        if not block_ids:
            return
        request = self.recompute.request(block_ids, reason="decode_requires_replay")
        self.recompute.recover(request)

    def after_decode_forward(self) -> None:
        self.decode_steps += 1
        self._collect_usage_this_forward = False
        resident_bytes = self.resident_bytes()
        live_pressure = self._classify_pressure_state(resident_bytes, emit_metrics=False)
        previous_pressure = self._pressure_state
        if (
            live_pressure is not PressureState.NORMAL
            or self._mutations_in_current_control_window > 0
        ):
            self._wake_active_control()
        mode = self._control_mode_for(
            resident_bytes,
            step=self.decode_steps,
            pressure=live_pressure,
        )
        interval_steps = self._control_interval_steps(mode)
        control_due = self.decode_steps - self._last_control_step >= interval_steps
        if live_pressure is not previous_pressure and live_pressure is not PressureState.NORMAL:
            control_due = True
        if not control_due:
            self._control_skipped_decode_steps_total += 1
            self.increment_perf("manager.control_skipped_decode_steps_total")
            self._pressure_state = live_pressure
            self._control_cadence_mode = mode
            return
        self._last_control_step = self.decode_steps
        self._control_windows_total += 1
        self.increment_perf("manager.control_windows_total")
        started = time.perf_counter()
        self._flush_usage_observers()
        prev_eval_ns = 0
        prev_host_ns = 0
        if self._adaptive_usage_timing_acc is not None:
            prev_eval_ns = self._adaptive_usage_timing_acc.get("eval_ns", 0)
            prev_host_ns = self._adaptive_usage_timing_acc.get("host_ns", 0)
        usage_started_ns = time.perf_counter_ns()
        usage = self.usage.snapshot_and_reset(timing_acc=self._adaptive_usage_timing_acc)
        self.record_perf_ns(
            "manager.usage_snapshot_total_ns",
            time.perf_counter_ns() - usage_started_ns,
        )
        if self._adaptive_usage_timing_acc is not None:
            self.record_perf_ns(
                "manager.usage_snapshot_eval_ns",
                self._adaptive_usage_timing_acc.get("eval_ns", 0) - prev_eval_ns,
            )
            self.record_perf_ns(
                "manager.usage_snapshot_host_ns",
                self._adaptive_usage_timing_acc.get("host_ns", 0) - prev_host_ns,
            )
        self._update_scores(usage)
        pressure = self._compute_pressure_state()
        self._update_hard_episode_state(pressure)
        protected = self._protected_block_ids()
        self._begin_cold_mutation_batch()
        try:
            self._apply_transitions(pressure=pressure, protected=protected)
            self._evict_if_needed(pressure=pressure, protected=protected)
        finally:
            self._end_cold_mutation_batch()
        self._emit_population()
        self._finalize_control_cadence(pressure=pressure)
        self.metrics.histogram(POLICY_TIME_SECONDS, time.perf_counter() - started)

    def should_sample_usage(self) -> bool:
        return self._collect_usage_this_forward

    def request_recompute(self, block_ids: tuple[int, ...], *, reason: str) -> Any:
        return self.recompute.request(block_ids, reason=reason)

    def record_decode_forward_time(self, seconds: float) -> None:
        if not self._recovery_wave_seen:
            return
        self.metrics.counter(POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL, seconds)
        self.metrics.counter(POST_RECOVERY_DECODE_FORWARDS_TOTAL)

    def attention_path_stats(self) -> dict[str, Any]:
        if not self._layer_caches:
            return {}
        rs = None
        last_error: str | None = None
        for layer_cache in self._layer_caches:
            try:
                rs = layer_cache.resident_state_for_execution()
                break
            except (AdaptiveKVError, RuntimeError) as exc:
                last_error = str(exc)
        if rs is None:
            return {"error": last_error or "no attention-bearing adaptive layer is available"}
        return {
            "n_execution_slabs": rs.n_execution_slabs,
            "n_execution_packs": rs.n_execution_packs,
            "pack_token_counts": list(rs.pack_token_counts),
            "n_visible_slices": len(rs.packs),
            "slab_token_counts": list(rs.slab_token_counts),
            "visible_spans": [pack_ref.visible_span for pack_ref in rs.packs],
            "fabric_compactions_total": rs.fabric_compactions_total,
            "execution_view_topology_rebuilds_total": rs.execution_view_topology_rebuilds_total,
            "observer_flushes_total": rs.observer_flushes_total,
        }

    def debug_snapshot(self) -> dict[str, Any]:
        blocks = self.registry.snapshot()
        return {
            "decode_steps": self.decode_steps,
            "pressure_state": self._pressure_state.value,
            "resident_bytes": self.resident_bytes(),
            "control_cadence": {
                "mode": self._control_cadence_mode.value,
                "base_interval_steps": self.config.update_window_steps,
                "effective_interval_steps": self._control_interval_steps(
                    self._control_cadence_mode
                ),
                "last_control_step": self._last_control_step,
                "calm_control_windows": self._calm_control_windows,
                "force_active_until_step": self._force_active_until_step,
                "control_windows_total": self._control_windows_total,
                "control_skipped_decode_steps_total": self._control_skipped_decode_steps_total,
                "usage_sample_steps_total": self._usage_sample_steps_total,
                "mutations_in_current_control_window": self._mutations_in_current_control_window,
            },
            "attention_path": self.attention_path_stats(),
            "performance_attribution": self._perf_trace.snapshot(),
            "hard_stabilization": {
                "episode_active": self._hard_episode_active,
                "stabilized_block_ids": sorted(self._hard_episode_stabilized_blocks),
                "recovery_hold_active": self._hard_episode_recovery_hold,
                "best_achievable_under_current_forward_semantics": self._hard_best_achievable,
                "reason": self._hard_best_achievable_reason,
                "over_budget_bytes": self._hard_best_achievable_over_budget_bytes,
                "blocking_block_ids": list(self._hard_best_achievable_blocking_block_ids),
            },
            "blocks": [
                block_debug_view(block, ghost_present=self.ghost_store.has(block.block_id))
                for block in blocks
            ],
            "ghosts": {block_id: ghost for block_id, ghost in self.ghost_store.snapshot().items()},
        }

    def resident_bytes(self) -> int:
        return sum(layer_cache.live_state_size_bytes for layer_cache in self._layer_caches)

    def history_token_count(self) -> int:
        if not self._layer_caches:
            return 0
        return self._layer_caches[0].offset

    def _required_evicted_block_ids(self) -> tuple[int, ...]:
        history_tokens = self.history_token_count()
        if history_tokens == 0:
            return ()
        required_start = min(
            (
                layer_cache.required_history_start(history_tokens)
                for layer_cache in self._layer_caches
            ),
            default=0,
        )
        return tuple(
            block.block_id
            for block in self.registry.required_evicted_blocks(
                history_tokens,
                start_token=required_start,
            )
        )

    def _protected_block_ids(self) -> set[int]:
        protected = self.registry.recent_tail_block_ids(self.config.recent_tail_protect_blocks)
        history_tokens = self.history_token_count()
        if history_tokens > 0:
            protected.add(self.registry.block_for_token(history_tokens - 1).block_id)
        return protected

    def _update_scores(self, usage: dict[int, float]) -> None:
        for block in self.registry.snapshot():
            updated = self.scoring.update(
                block,
                usage.get(block.block_id, 0.0),
                step=self.decode_steps,
            )
            self.registry.update(updated)
            self.metrics.counter(SCORE_UPDATES_TOTAL)

    def _compute_pressure_state(self) -> PressureState:
        return self._classify_pressure_state(self.resident_bytes(), emit_metrics=True)

    def _update_hard_episode_state(self, pressure: PressureState) -> None:
        if pressure is PressureState.HARD:
            if not self._hard_episode_active:
                self._hard_episode_active = True
                self._hard_episode_stabilized_blocks = set()
                self._hard_episode_recovery_hold = False
                self._clear_hard_best_achievable_state()
            return
        if self._hard_episode_active:
            self._hard_episode_active = False
            self._hard_episode_stabilized_blocks = set()
            self._hard_episode_recovery_hold = False
        self._clear_hard_best_achievable_state()

    def _apply_transitions(self, *, pressure: PressureState, protected: set[int]) -> None:
        for block in self.registry.snapshot():
            if self.transitions.should_restore(block, pressure=pressure, step=self.decode_steps):
                self._restore_block(block, reason="score_restore")
                continue
            if self.transitions.should_degrade(
                block,
                pressure=pressure,
                recent_tail=protected,
                step=self.decode_steps,
            ):
                self._degrade_block(block, reason="score_degrade")

    def _evict_if_needed(self, *, pressure: PressureState, protected: set[int]) -> None:
        if pressure is not PressureState.HARD:
            return
        candidates = self.evictions.select_candidates(
            self.registry.snapshot(),
            pressure=pressure,
            recent_tail=protected,
            avoid_block_ids=self._hard_eviction_avoid_block_ids(),
        )
        for block in candidates:
            current = self.registry.get(block.block_id)
            if not self.evictions.eligible(current, pressure=pressure):
                continue
            self._evict_block(current, reason="budget_hard")
            if self.config.hard_budget_bytes is None:
                break
            if self.resident_bytes() <= self.config.hard_budget_bytes:
                break
        self._update_hard_best_achievable_state(pressure=pressure, protected=protected)

    def _restore_block(self, block: BlockRecord, *, reason: str) -> None:
        if block.profile is ResidentProfile.TQ_SAFE:
            return
        if block.profile is ResidentProfile.EVICTED:
            raise AdaptiveKVError(
                f"Block {block.block_id} cannot restore from evicted state without recovery"
            )
        for cache in self._layer_caches:
            cache.restore_block(block.block_id)
        updated = self._mark_transition(
            block,
            to_profile=ResidentProfile.TQ_SAFE,
            reason=reason,
        )
        updated.last_restore_step = self.decode_steps
        self.registry.update(updated)
        self.ghost_store.mark_reactivated(block.block_id)
        self.metrics.counter(RESTORES_TOTAL)
        self._mutations_in_current_control_window += 1

    def _degrade_block(self, block: BlockRecord, *, reason: str) -> None:
        if block.profile is not ResidentProfile.TQ_SAFE:
            return
        for cache in self._layer_caches:
            cache.degrade_block(block.block_id)
        updated = self._mark_transition(
            block,
            to_profile=ResidentProfile.TQ_AGGR,
            reason=reason,
        )
        updated.last_degrade_step = self.decode_steps
        self.registry.update(updated)
        self.metrics.counter(DEGRADES_TOTAL)
        self._mutations_in_current_control_window += 1

    def _evict_block(self, block: BlockRecord, *, reason: str) -> None:
        if block.pin_state is PinState.HARD:
            return
        if block.profile is not ResidentProfile.TQ_AGGR:
            raise AdaptiveKVError(
                "Adaptive hard eviction requires TQ_AGGR resident state; "
                f"block {block.block_id} is {block.profile.value}"
            )
        for cache in self._layer_caches:
            cache.evict_block(block.block_id)
        updated = self._mark_transition(
            block,
            to_profile=ResidentProfile.EVICTED,
            reason=reason,
        )
        self.registry.update(updated)
        self.ghost_store.create(updated, step=self.decode_steps)
        self.metrics.counter(EVICTIONS_TOTAL)
        self._mutations_in_current_control_window += 1

    def _recover_request(self, request: Any) -> None:
        if self._model is None:
            raise AdaptiveKVUnsupportedError(
                "Adaptive KV recovery requires a bound generation model runtime"
            )
        history_tokens = self.history_token_count()
        if history_tokens <= 0:
            return
        replay_tokens = self._replay_token_count_for_request(request, history_tokens)
        if replay_tokens <= 0:
            return
        scratch = self._ensure_scratch_replay_prefix(replay_tokens)
        recovered_any = False
        materialize_started = time.perf_counter()
        self._begin_cold_mutation_batch()
        try:
            for recovery_group in self._recovery_groups(request.block_ids):
                for layer_cache, scratch_layer in zip(self._layer_caches, scratch, strict=True):
                    layer_cache.recover_blocks_from_scratch(
                        recovery_group,
                        scratch_layer,
                        self.replay_backend,
                        recovery_profile=ResidentProfile.TQ_AGGR,
                    )
                for block in recovery_group:
                    updated = self._mark_transition(
                        block,
                        to_profile=ResidentProfile.TQ_AGGR,
                        reason="recovered_replay",
                    )
                    self.registry.update(updated)
                    self.ghost_store.mark_reactivated(block.block_id)
                    recovered_any = True
                    if self._hard_episode_active:
                        self._hard_episode_stabilized_blocks.add(block.block_id)
        finally:
            self._end_cold_mutation_batch()
        if recovered_any:
            self.metrics.counter(
                RECOVERY_MATERIALIZATION_TIME_SECONDS_TOTAL,
                time.perf_counter() - materialize_started,
            )
            self.metrics.counter(RECOVERY_MATERIALIZATION_EVENTS_TOTAL)
            self._recovery_wave_seen = True
            self._wake_active_control()
        if recovered_any and self._hard_episode_active:
            self._hard_episode_recovery_hold = True

    @staticmethod
    def _replay_token_count_for_request(request: Any, history_tokens: int) -> int:
        if history_tokens <= 0:
            return 0
        needed = max((end for _, end in request.source_spans), default=0)
        return min(history_tokens, needed)

    def _recovery_groups(self, block_ids: tuple[int, ...]) -> tuple[tuple[BlockRecord, ...], ...]:
        groups: list[list[BlockRecord]] = []
        current: list[BlockRecord] = []
        for block_id in block_ids:
            block = self.registry.get(block_id)
            if block.profile is not ResidentProfile.EVICTED:
                continue
            if current and current[-1].end_token != block.start_token:
                groups.append(current)
                current = []
            current.append(block)
        if current:
            groups.append(current)
        return tuple(tuple(group) for group in groups)

    def _begin_cold_mutation_batch(self) -> None:
        for cache in self._layer_caches:
            begin = getattr(cache, "begin_cold_mutation_batch", None)
            if begin is not None:
                begin()

    def _end_cold_mutation_batch(self) -> None:
        for cache in reversed(self._layer_caches):
            end = getattr(cache, "end_cold_mutation_batch", None)
            if end is not None:
                end()

    def _flush_usage_observers(self) -> None:
        for cache in self._layer_caches:
            flush = getattr(cache, "flush_usage_observer", None)
            if flush is not None:
                flush()

    def _ensure_scratch_replay_prefix(self, total_tokens: int) -> list[Any]:
        previous_replayed_tokens = self._scratch_replayed_tokens
        replay_started = time.perf_counter()
        replay_state = self.replay_backend.ensure_scratch_replay_prefix(
            model=self._model,
            num_layers=self._num_layers,
            scratch_cache=self._scratch_replay_cache,
            replayed_tokens=self._scratch_replayed_tokens,
            materialized=self._scratch_replay_materialized,
            source_tokens=self.source_tokens,
            total_tokens=total_tokens,
            prefill_step_size=self._prefill_step_size,
        )
        self._scratch_replay_cache = replay_state.cache
        self._scratch_replayed_tokens = replay_state.replayed_tokens
        self._scratch_replay_materialized = replay_state.materialized
        if replay_state.replayed_tokens <= previous_replayed_tokens:
            return replay_state.cache
        self.metrics.counter(
            REPLAY_FORWARD_TIME_SECONDS_TOTAL,
            time.perf_counter() - replay_started,
        )
        self.metrics.counter(REPLAY_FORWARD_EVENTS_TOTAL)
        return replay_state.cache

    def _reset_scratch_replay(self) -> None:
        self._scratch_replay_cache = None
        self._scratch_replayed_tokens = 0
        self._scratch_replay_materialized = True

    def _mark_transition(
        self,
        block: BlockRecord,
        *,
        to_profile: ResidentProfile,
        reason: str,
    ) -> BlockRecord:
        transition = TransitionRecord(
            step=self.decode_steps,
            from_profile=block.profile,
            to_profile=to_profile,
            reason=reason,
        )
        return BlockRecord(
            block_id=block.block_id,
            start_token=block.start_token,
            end_token=block.end_token,
            source_start=block.source_start,
            source_end=block.source_end,
            segment_id=block.segment_id,
            pin_state=block.pin_state,
            profile=to_profile,
            created_step=block.created_step,
            structural_prior=block.structural_prior,
            age_windows=block.age_windows,
            windows_in_profile=0,
            last_access_step=block.last_access_step,
            last_transition_step=self.decode_steps,
            last_restore_step=block.last_restore_step,
            last_degrade_step=block.last_degrade_step,
            score=block.score,
            last_transition=transition,
        )

    def _emit_population(self) -> None:
        if not self.config.emit_metrics:
            return
        emit_population(self.metrics, self.registry.snapshot())

    def _classify_pressure_state(
        self,
        resident_bytes: int,
        *,
        emit_metrics: bool,
    ) -> PressureState:
        if (
            self.config.hard_budget_bytes is not None
            and resident_bytes >= self.config.hard_budget_bytes
        ):
            if emit_metrics:
                self.metrics.counter(PRESSURE_HARD_COUNT)
            pressure = PressureState.HARD
        elif (
            self.config.soft_budget_bytes is not None
            and resident_bytes >= self.config.soft_budget_bytes
        ):
            if emit_metrics:
                self.metrics.counter(PRESSURE_SOFT_COUNT)
            pressure = PressureState.SOFT
        else:
            pressure = PressureState.NORMAL
        self._pressure_state = pressure
        return pressure

    def _near_soft_budget(self, resident_bytes: int) -> bool:
        soft_budget = self.config.soft_budget_bytes
        if soft_budget is None:
            return False
        wake_threshold = max(1, int(soft_budget * _SOFT_WAKE_FRACTION))
        return resident_bytes >= wake_threshold

    def _control_interval_steps(self, mode: _ControlCadenceMode) -> int:
        base = self.config.update_window_steps
        if base <= 1:
            return 1
        if mode is _ControlCadenceMode.DORMANT:
            return base * _DORMANT_INTERVAL_MULTIPLIER
        return base

    def _control_mode_for(
        self,
        resident_bytes: int,
        *,
        step: int,
        pressure: PressureState | None = None,
    ) -> _ControlCadenceMode:
        current_pressure = (
            pressure
            if pressure is not None
            else self._classify_pressure_state(resident_bytes, emit_metrics=False)
        )
        if current_pressure is PressureState.HARD:
            return _ControlCadenceMode.STRESSED
        if current_pressure is PressureState.SOFT:
            return _ControlCadenceMode.ACTIVE
        if self._near_soft_budget(resident_bytes):
            return _ControlCadenceMode.ACTIVE
        if step <= self._force_active_until_step:
            return _ControlCadenceMode.ACTIVE
        if self._calm_control_windows >= _DORMANT_ENTRY_CALM_WINDOWS:
            return _ControlCadenceMode.DORMANT
        return _ControlCadenceMode.ACTIVE

    def _wake_active_control(self) -> None:
        base = max(1, self.config.update_window_steps)
        self._force_active_until_step = max(
            self._force_active_until_step,
            self.decode_steps + (base * _POST_MUTATION_ACTIVE_WINDOWS),
        )

    def _finalize_control_cadence(self, *, pressure: PressureState) -> None:
        resident_bytes = self.resident_bytes()
        if (
            pressure is PressureState.NORMAL
            and not self._near_soft_budget(resident_bytes)
            and self._mutations_in_current_control_window == 0
            and self._pending_recompute_requests == 0
        ):
            self._calm_control_windows += 1
        else:
            self._calm_control_windows = 0
            if (
                self._mutations_in_current_control_window > 0
                or self._pending_recompute_requests > 0
                or pressure is not PressureState.NORMAL
            ):
                self._wake_active_control()
        self._control_cadence_mode = self._control_mode_for(
            resident_bytes,
            step=self.decode_steps,
            pressure=pressure,
        )
        self._mutations_in_current_control_window = 0
        self._pending_recompute_requests = 0

    def _clear_hard_best_achievable_state(self) -> None:
        self._hard_best_achievable = False
        self._hard_best_achievable_reason = None
        self._hard_best_achievable_over_budget_bytes = 0
        self._hard_best_achievable_blocking_block_ids = ()

    def _hard_eviction_avoid_block_ids(self) -> set[int]:
        avoid = set(self._hard_episode_stabilized_blocks)
        if not self._hard_episode_recovery_hold:
            return avoid
        history_tokens = self.history_token_count()
        if history_tokens <= 0:
            return avoid
        avoid.update(
            block.block_id
            for block in self.registry.snapshot()
            if block.profile is ResidentProfile.TQ_AGGR and block.start_token < history_tokens
        )
        return avoid

    def _update_hard_best_achievable_state(
        self,
        *,
        pressure: PressureState,
        protected: set[int],
    ) -> None:
        self._clear_hard_best_achievable_state()
        if pressure is not PressureState.HARD or self.config.hard_budget_bytes is None:
            return
        resident_bytes = self.resident_bytes()
        if resident_bytes <= self.config.hard_budget_bytes:
            return
        avoid_block_ids = self._hard_eviction_avoid_block_ids()
        remaining_candidates = self.evictions.select_candidates(
            self.registry.snapshot(),
            pressure=pressure,
            recent_tail=protected,
            avoid_block_ids=avoid_block_ids,
        )
        if remaining_candidates:
            return
        blocking_blocks = [
            block.block_id
            for block in self.registry.snapshot()
            if block.profile is not ResidentProfile.EVICTED
            and (
                block.pin_state is PinState.HARD
                or block.block_id in protected
                or block.block_id in avoid_block_ids
            )
        ]
        if not blocking_blocks:
            return
        self._hard_best_achievable = True
        if self._hard_episode_stabilized_blocks:
            self._hard_best_achievable_reason = "required_history_recovered_under_hard_episode"
        else:
            self._hard_best_achievable_reason = "only_protected_blocks_remain"
        self._hard_best_achievable_over_budget_bytes = (
            resident_bytes - self.config.hard_budget_bytes
        )
        self._hard_best_achievable_blocking_block_ids = tuple(sorted(blocking_blocks))

    def _increment_recompute_requests(self) -> None:
        self._pending_recompute_requests += 1
        self.metrics.counter(RECOMPUTATIONS_TOTAL)
        self._wake_active_control()

    def _should_sample_usage(self) -> bool:
        next_step = self.decode_steps + 1
        if self.config.update_window_steps <= 1:
            return True
        resident_bytes = self.resident_bytes()
        pressure = self._classify_pressure_state(resident_bytes, emit_metrics=False)
        mode = self._control_mode_for(resident_bytes, step=next_step, pressure=pressure)
        if next_step - self._last_control_step >= self._control_interval_steps(mode):
            return True
        if pressure is not PressureState.NORMAL:
            return True
        return self._near_soft_budget(resident_bytes)
