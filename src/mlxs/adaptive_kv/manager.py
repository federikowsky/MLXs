"""Adaptive KV manager and cache wrapper."""

from __future__ import annotations

import time
from typing import Any

from mlxs.adaptive_kv.adapters import default_generation_adapter
from mlxs.adaptive_kv.block_registry import AdaptiveBlockRegistry
from mlxs.adaptive_kv.block_types import (
    BlockRecord,
    BlockTier,
    PinState,
    PressureState,
    TransitionRecord,
)
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.eviction import AdaptiveEvictionEngine
from mlxs.adaptive_kv.exceptions import AdaptiveKVError, AdaptiveKVUnsupportedError
from mlxs.adaptive_kv.ghost import AdaptiveGhostStore
from mlxs.adaptive_kv.metrics import (
    DEMOTIONS_TOTAL,
    EVICTIONS_TOTAL,
    POLICY_TIME_SECONDS,
    POST_RECOVERY_DECODE_FORWARDS_TOTAL,
    POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL,
    PRESSURE_HARD_COUNT,
    PRESSURE_SOFT_COUNT,
    PROMOTIONS_TOTAL,
    RECOMPUTATIONS_TOTAL,
    RECOVERY_MATERIALIZATION_EVENTS_TOTAL,
    RECOVERY_MATERIALIZATION_TIME_SECONDS_TOTAL,
    REPLAY_FORWARD_EVENTS_TOTAL,
    REPLAY_FORWARD_TIME_SECONDS_TOTAL,
    SCORE_UPDATES_TOTAL,
    block_debug_view,
    emit_population,
)
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

# Backward-compatible alias retained for tests and diagnostics that still refer to
# the currently retained concrete layer runtime.
AdaptiveLayerCache = _DEFAULT_LAYER_RUNTIME


class AdaptiveKVManager:
    """Top-level coordinator for adaptive KV V1."""

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
        self._model: Any = None
        self._prefill_step_size = 2048
        self._pending_recompute_requests = 0
        self._adaptive_usage_timing_acc: dict[str, int] | None = None
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

    def bump_resident_version(self) -> None:
        self._resident_version += 1

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

    def ensure_required_resident(self) -> None:
        block_ids = self._required_evicted_block_ids()
        if not block_ids:
            return
        request = self.recompute.request(block_ids, reason="decode_requires_replay")
        self.recompute.recover(request)

    def after_decode_forward(self) -> None:
        self.decode_steps += 1
        self._collect_usage_this_forward = False
        if self.decode_steps % self.config.update_window_steps != 0:
            return
        started = time.perf_counter()
        usage = self.usage.snapshot_and_reset(timing_acc=self._adaptive_usage_timing_acc)
        self._update_scores(usage)
        pressure = self._compute_pressure_state()
        self._update_hard_episode_state(pressure)
        protected = self._protected_block_ids()
        self._apply_transitions(pressure=pressure, protected=protected)
        self._evict_if_needed(pressure=pressure, protected=protected)
        self._emit_population()
        self.metrics.histogram(POLICY_TIME_SECONDS, time.perf_counter() - started)

    def should_sample_usage(self) -> bool:
        if self._collect_usage_this_forward:
            return True
        return any(block.tier is BlockTier.COMPRESSED for block in self.registry.snapshot())

    def force_full_for_append(self, block_id: int) -> BlockRecord:
        block = self.registry.get(block_id)
        if block.tier is BlockTier.FULL:
            return block
        if block.tier is BlockTier.EVICTED:
            raise AdaptiveKVError(
                f"Block {block_id} cannot be resurrected for append without replay"
            )
        self._promote_block(block, reason="append_requires_full")
        return self.registry.get(block_id)

    def request_recompute(self, block_ids: tuple[int, ...], *, reason: str) -> Any:
        return self.recompute.request(block_ids, reason=reason)

    def record_decode_forward_time(self, seconds: float) -> None:
        if not self._recovery_wave_seen:
            return
        self.metrics.counter(POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL, seconds)
        self.metrics.counter(POST_RECOVERY_DECODE_FORWARDS_TOTAL)

    def attention_path_stats(self) -> dict[str, Any]:
        """Diagnostic only: segment structure for a representative attention-bearing layer.

        Counts match ``AdaptiveLayerCache.resident_state_for_attention`` — the hot-path
        segment list (FULL runs coalesced; each compressed *run* is one segment).
        """
        if not self._layer_caches:
            return {}
        rs = None
        last_error: str | None = None
        for layer_cache in self._layer_caches:
            if not layer_cache.should_sample_usage():
                continue
            try:
                rs = layer_cache.resident_state_for_attention()
                break
            except (AdaptiveKVError, RuntimeError) as exc:
                last_error = str(exc)
        if rs is None:
            return {"error": last_error or "no attention-bearing adaptive layer is available"}
        n_full = sum(1 for s in rs.segments if s.tier is BlockTier.FULL)
        n_comp = sum(1 for s in rs.segments if s.tier is BlockTier.COMPRESSED)
        logical_in_comp_runs = sum(
            len(s.block_slices) for s in rs.segments if s.tier is BlockTier.COMPRESSED
        )
        logical_in_full_segments = sum(
            len(s.block_slices) for s in rs.segments if s.tier is BlockTier.FULL
        )
        comp_counts = [s.token_count for s in rs.segments if s.tier is BlockTier.COMPRESSED]
        comp_span = sum(comp_counts)
        return {
            "n_attention_segments": len(rs.segments),
            "n_full_attention_segments": n_full,
            "n_compressed_attention_segments": n_comp,
            "logical_blocks_in_compressed_runs": logical_in_comp_runs,
            "logical_blocks_in_full_segments": logical_in_full_segments,
            "compressed_attention_token_span": comp_span,
            "max_compressed_segment_tokens": max(comp_counts) if comp_counts else 0,
            "compressed_segment_token_counts": comp_counts,
        }

    def debug_snapshot(self) -> dict[str, Any]:
        blocks = self.registry.snapshot()
        return {
            "decode_steps": self.decode_steps,
            "pressure_state": self._pressure_state.value,
            "resident_bytes": self.resident_bytes(),
            "attention_path": self.attention_path_stats(),
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
        return tuple(
            block.block_id for block in self.registry.required_evicted_blocks(history_tokens)
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
        resident_bytes = self.resident_bytes()
        if (
            self.config.hard_budget_bytes is not None
            and resident_bytes >= self.config.hard_budget_bytes
        ):
            self.metrics.counter(PRESSURE_HARD_COUNT)
            self._pressure_state = PressureState.HARD
        elif (
            self.config.soft_budget_bytes is not None
            and resident_bytes >= self.config.soft_budget_bytes
        ):
            self.metrics.counter(PRESSURE_SOFT_COUNT)
            self._pressure_state = PressureState.SOFT
        else:
            self._pressure_state = PressureState.NORMAL
        return self._pressure_state

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
            if self.transitions.should_promote(block, pressure=pressure, step=self.decode_steps):
                self._promote_block(block, reason="score_promote")
                continue
            if self.transitions.should_demote(
                block,
                pressure=pressure,
                recent_tail=protected,
                step=self.decode_steps,
            ):
                self._demote_block(block, reason="score_demote")

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

    def _promote_block(self, block: BlockRecord, *, reason: str) -> None:
        if block.tier is BlockTier.FULL:
            return
        if block.tier is BlockTier.EVICTED:
            raise AdaptiveKVError(
                f"Block {block.block_id} cannot promote from EVICTED without recovery"
            )
        for cache in self._layer_caches:
            cache.promote_block(block.block_id)
        updated = self._mark_transition(block, to_tier=BlockTier.FULL, reason=reason)
        updated.last_promote_step = self.decode_steps
        self.registry.update(updated)
        self.ghost_store.mark_reactivated(block.block_id)
        self.metrics.counter(PROMOTIONS_TOTAL)

    def _demote_block(self, block: BlockRecord, *, reason: str) -> None:
        if block.tier is not BlockTier.FULL:
            return
        for cache in self._layer_caches:
            cache.demote_block(block.block_id)
        updated = self._mark_transition(block, to_tier=BlockTier.COMPRESSED, reason=reason)
        updated.last_demote_step = self.decode_steps
        self.registry.update(updated)
        self.metrics.counter(DEMOTIONS_TOTAL)

    def _evict_block(self, block: BlockRecord, *, reason: str) -> None:
        if block.pin_state is PinState.HARD:
            return
        if block.tier is not BlockTier.COMPRESSED:
            raise AdaptiveKVError(
                f"Adaptive hard eviction requires COMPRESSED tier; block {block.block_id} is "
                f"{block.tier.value}"
            )
        for cache in self._layer_caches:
            cache.evict_block(block.block_id)
        updated = self._mark_transition(block, to_tier=BlockTier.EVICTED, reason=reason)
        self.registry.update(updated)
        self.ghost_store.create(updated, step=self.decode_steps)
        self.metrics.counter(EVICTIONS_TOTAL)

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
        for recovery_group in self._recovery_groups(request.block_ids):
            for layer_cache, scratch_layer in zip(self._layer_caches, scratch, strict=True):
                layer_cache.recover_blocks_from_scratch(
                    recovery_group,
                    scratch_layer,
                    self.replay_backend,
                )
            for block in recovery_group:
                updated = self._mark_transition(
                    block,
                    to_tier=BlockTier.COMPRESSED,
                    reason="recovered_replay",
                )
                self.registry.update(updated)
                self.ghost_store.mark_reactivated(block.block_id)
                recovered_any = True
                if self._hard_episode_active:
                    self._hard_episode_stabilized_blocks.add(block.block_id)
        if recovered_any:
            self.metrics.counter(
                RECOVERY_MATERIALIZATION_TIME_SECONDS_TOTAL,
                time.perf_counter() - materialize_started,
            )
            self.metrics.counter(RECOVERY_MATERIALIZATION_EVENTS_TOTAL)
            self._recovery_wave_seen = True
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
            if block.tier is not BlockTier.EVICTED:
                continue
            if current and current[-1].end_token != block.start_token:
                groups.append(current)
                current = []
            current.append(block)
        if current:
            groups.append(current)
        return tuple(tuple(group) for group in groups)

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
        to_tier: BlockTier,
        reason: str,
    ) -> BlockRecord:
        transition = TransitionRecord(
            step=self.decode_steps,
            from_tier=block.tier,
            to_tier=to_tier,
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
            tier=to_tier,
            created_step=block.created_step,
            structural_prior=block.structural_prior,
            age_windows=block.age_windows,
            windows_in_tier=0,
            last_access_step=block.last_access_step,
            last_transition_step=self.decode_steps,
            last_promote_step=block.last_promote_step,
            last_demote_step=block.last_demote_step,
            score=block.score,
            last_transition=transition,
        )

    def _emit_population(self) -> None:
        if not self.config.emit_metrics:
            return
        emit_population(self.metrics, self.registry.snapshot())

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
            if block.tier is BlockTier.COMPRESSED and block.start_token < history_tokens
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
            if block.tier is not BlockTier.EVICTED
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

    def _should_sample_usage(self) -> bool:
        next_step = self.decode_steps + 1
        if self.config.update_window_steps <= 1:
            return True
        return next_step % self.config.update_window_steps == 0
