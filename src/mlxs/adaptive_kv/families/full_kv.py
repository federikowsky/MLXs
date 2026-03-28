"""Family A runtime substrate: TurboQuant-first full-history KV decoders."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from mlxs.adaptive_kv.block_types import BlockRecord, ResidentProfile
from mlxs.adaptive_kv.exceptions import AdaptiveKVError
from mlxs.adaptive_kv.resident import (
    ResidentBlockHandle,
    ResidentExecutionMode,
    TurboQuantResidentBackend,
)
from mlxs.adaptive_kv.runtime import (
    AdaptiveKVReplayBackend,
    AdaptiveKVRuntimeSubstrate,
    RuntimeFamily,
    RuntimeFamilyBindings,
    RuntimeFamilyDescriptor,
    ScratchReplayState,
)
from mlxs.adaptive_kv.storage import ResidentStateView
from mlxs.cache.attention_mask import _mask_from_length
from mlxs.cache.kv import KVCache

if TYPE_CHECKING:
    from mlxs.adaptive_kv.manager import AdaptiveKVManager


FULL_KV_FAMILY = RuntimeFamilyDescriptor(
    family=RuntimeFamily.FULL_KV,
    display_name="TurboQuant Full-KV Decoder",
    summary=(
        "Token-addressable TurboQuant resident history with resident profiles "
        "TQ_SAFE/TQ_AGGR and replay-backed recovery."
    ),
    token_addressable=True,
)


class FullAttentionKVRuntimeSubstrate(AdaptiveKVRuntimeSubstrate):
    """Runtime substrate for TurboQuant-first token-addressable full-KV families."""

    def make_layer_runtime(
        self,
        manager: AdaptiveKVManager,
        layer_index: int,
    ) -> FullAttentionKVAdaptiveLayerCache:
        backend = TurboQuantResidentBackend(
            safe_bits=manager.config.tq_safe_bits,
            aggr_bits=manager.config.tq_aggr_bits,
            safe_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
            aggr_execution_mode=self._aggr_execution_mode(),
            perf_trace=manager.perf_trace,
        )
        return FullAttentionKVAdaptiveLayerCache(
            manager,
            layer_index=layer_index,
            backend=backend,
        )

    def _aggr_execution_mode(self) -> ResidentExecutionMode:
        return ResidentExecutionMode.DEQUANTIZE_ON_READ


class FullAttentionKVReplayBackend(AdaptiveKVReplayBackend):
    """Replay / recovery backend for token-addressable full-KV models."""

    @staticmethod
    def _cache_eval_tensors(cache: Any) -> list[mx.array]:
        state = getattr(cache, "state", None)
        if state is not None:
            if isinstance(state, tuple):
                return [tensor for tensor in state if hasattr(tensor, "nbytes")]
            if hasattr(state, "nbytes"):
                return [state]
        state_list = getattr(cache, "cache", None)
        if isinstance(state_list, list):
            return [tensor for tensor in state_list if hasattr(tensor, "nbytes")]
        return []

    def ensure_scratch_replay_prefix(
        self,
        *,
        model: Any,
        num_layers: int,
        scratch_cache: list[Any] | None,
        replayed_tokens: int,
        materialized: bool,
        source_tokens: list[int],
        total_tokens: int,
        prefill_step_size: int,
    ) -> ScratchReplayState:
        total_tokens = max(0, min(total_tokens, len(source_tokens)))
        if scratch_cache is None:
            scratch_cache = [KVCache() for _ in range(num_layers)]
            replayed_tokens = 0
            materialized = True
        if total_tokens <= replayed_tokens:
            return ScratchReplayState(
                cache=scratch_cache,
                replayed_tokens=replayed_tokens,
                materialized=materialized,
            )
        offset = replayed_tokens
        if not materialized and offset > 0:
            tensors = [
                tensor
                for cache_entry in scratch_cache
                for tensor in self._cache_eval_tensors(cache_entry)
            ]
            if tensors:
                mx.eval(tensors)
            materialized = True
        while offset < total_tokens:
            n = min(prefill_step_size, total_tokens - offset)
            chunk = mx.array(source_tokens[offset : offset + n])
            model(chunk[None], cache=scratch_cache)
            offset += n
            if offset < total_tokens:
                tensors = [
                    tensor
                    for cache_entry in scratch_cache
                    for tensor in self._cache_eval_tensors(cache_entry)
                ]
                if tensors:
                    mx.eval(tensors)
                materialized = True
                mx.clear_cache()
            else:
                materialized = False
        return ScratchReplayState(
            cache=scratch_cache,
            replayed_tokens=total_tokens,
            materialized=materialized,
        )

    def copy_replay_token_range(
        self,
        replay_layer: Any,
        start_token: int,
        end_token: int,
    ) -> tuple[mx.array, mx.array]:
        return replay_layer.copy_token_range(start_token, end_token)


class FullAttentionKVAdaptiveLayerCache:
    """Per-layer TurboQuant-first adaptive cache wrapper for full-KV runtimes."""

    def __init__(
        self,
        manager: AdaptiveKVManager,
        *,
        layer_index: int,
        backend: TurboQuantResidentBackend,
    ) -> None:
        self._manager = manager
        self._layer_index = layer_index
        self._backend = backend
        self._logical_offset = 0
        self._handles: dict[int, ResidentBlockHandle] = {}
        self._resident_state: ResidentStateView | None = None
        self._topology_epoch = 0
        self._tail_epoch = 0
        self._resident_view_topology_epoch = -1
        self._resident_view_tail_epoch = -1
        self._resident_view_query_key: Any = None
        self._resident_view_visible_start = 0
        self._execution_view_topology_rebuilds_total = 0
        self._execution_view_local_repairs_total = 0
        self._execution_view_repaired_suffix_tokens_total = 0
        self._dtype: Any = None
        self._cold_batch_depth = 0
        self._cold_topology_dirty = False
        self._append_only_topology_pending = False
        self._append_only_repair_logical_start: int | None = None

    @property
    def offset(self) -> int:
        return self._logical_offset

    @property
    def keys(self) -> mx.array | None:
        tensors = self._debug_arrays()
        return tensors[0]

    @property
    def values(self) -> mx.array | None:
        tensors = self._debug_arrays()
        return tensors[1]

    @property
    def state(self) -> tuple[Any, ...] | None:
        ordered = self._ordered_handles(self._logical_offset)
        if not ordered:
            return None
        tensors: list[Any] = []
        seen: set[int] = set()
        for slab in self._backend.fabric.slabs_for_handles(tuple(ordered)):
            if slab.slab_id in seen:
                continue
            seen.add(slab.slab_id)
            tensors.extend((slab.keys, slab.values))
        return tuple(tensors)

    @property
    def live_state_size_bytes(self) -> int:
        return sum(handle.live_bytes for handle in self._handles.values())

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[Any, Any]:
        total_started_ns = time.perf_counter_ns()
        start = self._logical_offset
        end = start + keys.shape[2]
        if self._dtype is None:
            self._dtype = keys.dtype
        self._manager.ensure_block_coverage(end)
        append_started_ns = time.perf_counter_ns()
        tail_epoch_changed = False
        append_only_topology_changed = False
        repair_logical_start: int | None = None
        for block_id, local_start, local_end in self._manager.registry.token_slices(start, end):
            block = self._manager.registry.get(block_id)
            if block.profile is ResidentProfile.EVICTED:
                raise AdaptiveKVError(
                    f"Cannot append into evicted block {block.block_id} without recovery"
                )
            tail_only = self._append_to_block(
                block,
                keys[..., local_start:local_end, :],
                values[..., local_start:local_end, :],
            )
            if tail_only:
                tail_epoch_changed = True
            else:
                append_only_topology_changed = True
                repair_logical_start = (
                    block.start_token
                    if repair_logical_start is None
                    else min(repair_logical_start, block.start_token)
                )
        self.record_perf_ns(
            "cache.update_and_fetch_append_ns",
            time.perf_counter_ns() - append_started_ns,
        )
        if append_only_topology_changed:
            self._topology_epoch += 1
            self._append_only_topology_pending = True
            self._append_only_repair_logical_start = repair_logical_start
        if tail_epoch_changed:
            self._tail_epoch += 1
        self._logical_offset = end
        self._manager.bump_resident_version()
        resident_state = self.resident_state_for_execution(query_tokens=keys.shape[2])
        self.record_perf_ns(
            "cache.update_and_fetch_total_ns",
            time.perf_counter_ns() - total_started_ns,
        )
        return resident_state, resident_state

    def remove_token_range(self, start: int, end: int) -> None:
        if end <= start:
            return
        removals = [
            block.block_id
            for block in self._manager.registry.snapshot()
            if block.resident and not (block.end_token <= start or block.start_token >= end)
        ]
        self.begin_cold_mutation_batch()
        try:
            for block_id in removals:
                handle = self._handles.pop(block_id, None)
                if handle is not None:
                    self._backend.evict_handle(handle)
        finally:
            self.end_cold_mutation_batch()
        if removals:
            self._mark_topology_change()
            self._manager.bump_resident_version()

    def degrade_block(self, block_id: int) -> None:
        handle = self._handles.get(block_id)
        if handle is None or handle.profile is not ResidentProfile.TQ_SAFE:
            return
        self._handles[block_id] = self._backend.convert_profile(
            handle,
            profile=ResidentProfile.TQ_AGGR,
            dtype=self._effective_dtype(),
        )
        self._mark_topology_change()
        self._manager.bump_resident_version()

    def restore_block(self, block_id: int) -> None:
        handle = self._handles.get(block_id)
        if handle is None or handle.profile is not ResidentProfile.TQ_AGGR:
            return
        self._handles[block_id] = self._backend.convert_profile(
            handle,
            profile=ResidentProfile.TQ_SAFE,
            dtype=self._effective_dtype(),
        )
        self._mark_topology_change()
        self._manager.bump_resident_version()

    def evict_block(self, block_id: int) -> None:
        handle = self._handles.pop(block_id, None)
        if handle is not None:
            self._backend.evict_handle(handle)
            self._mark_topology_change()
        self._manager.bump_resident_version()

    def recover_blocks_from_scratch(
        self,
        blocks: tuple[BlockRecord, ...],
        replay_layer: Any,
        replay_backend: AdaptiveKVReplayBackend,
        *,
        recovery_profile: ResidentProfile,
    ) -> None:
        dtype = self._effective_dtype()
        recovered_any = False
        self.begin_cold_mutation_batch()
        try:
            for block in blocks:
                keys, values = replay_backend.copy_replay_token_range(
                    replay_layer,
                    block.source_start,
                    block.source_end,
                )
                self._handles[block.block_id] = self._backend.create_handle(
                    block_id=block.block_id,
                    profile=recovery_profile,
                    logical_span=(block.start_token, block.end_token),
                    keys=keys.astype(dtype),
                    values=values.astype(dtype),
                )
                recovered_any = True
        finally:
            self.end_cold_mutation_batch()
        if recovered_any:
            self._mark_topology_change()
        self._manager.bump_resident_version()

    def resident_state_for_execution(self, *, query_tokens: int = 1) -> ResidentStateView:
        total_started_ns = time.perf_counter_ns()
        query_key_started_ns = total_started_ns
        query_key = self._resident_view_query_key_for(query_tokens=query_tokens)
        self.record_perf_ns(
            "view.query_key_ns",
            time.perf_counter_ns() - query_key_started_ns,
        )
        visible_started_ns = time.perf_counter_ns()
        visible_start = self._resident_visible_start_for_query(query_tokens=query_tokens)
        self.record_perf_ns(
            "view.visible_start_ns",
            time.perf_counter_ns() - visible_started_ns,
        )
        if (
            self._resident_state is not None
            and self._resident_view_topology_epoch == self._topology_epoch
            and self._resident_view_tail_epoch == self._tail_epoch
            and self._resident_view_query_key == query_key
            and self._resident_view_visible_start == visible_start
        ):
            self.increment_perf("view.cache_hit_count")
            self.record_perf_ns(
                "view.resident_state_total_ns",
                time.perf_counter_ns() - total_started_ns,
            )
            return self._resident_state

        self.increment_perf("view.cache_miss_count")
        ordered_started_ns = time.perf_counter_ns()
        ordered = tuple(self._ordered_handles(self._logical_offset))
        self.record_perf_ns(
            "view.ordered_handles_ns",
            time.perf_counter_ns() - ordered_started_ns,
        )
        if self._resident_view_topology_epoch != self._topology_epoch:
            self._execution_view_topology_rebuilds_total += 1
        backend_started_ns = time.perf_counter_ns()
        queried = self._query_resident_state(ordered, query_tokens=query_tokens)
        self.record_perf_ns(
            "view.backend_query_ns",
            time.perf_counter_ns() - backend_started_ns,
        )
        compose_started_ns = time.perf_counter_ns()
        self._resident_state = self._compose_resident_state(queried.slices)
        self.record_perf_ns(
            "view.compose_state_ns",
            time.perf_counter_ns() - compose_started_ns,
        )
        self._resident_view_topology_epoch = self._topology_epoch
        self._resident_view_tail_epoch = self._tail_epoch
        self._resident_view_query_key = query_key
        self._resident_view_visible_start = visible_start
        self._append_only_topology_pending = False
        self._append_only_repair_logical_start = None
        self.record_perf_ns(
            "view.resident_state_total_ns",
            time.perf_counter_ns() - total_started_ns,
        )
        return self._resident_state

    def _resident_view_query_key_for(self, *, query_tokens: int) -> Any:
        del query_tokens
        return "full"

    def _resident_visible_start_for_query(self, *, query_tokens: int) -> int:
        del query_tokens
        return 0

    def _query_resident_state(
        self,
        ordered: tuple[ResidentBlockHandle, ...],
        *,
        query_tokens: int,
    ) -> ResidentStateView:
        del query_tokens
        return self._backend.query_full_view(
            ordered,
            topology_epoch=self._topology_epoch,
            tail_epoch=self._tail_epoch,
            execution_view_topology_rebuilds_total=self._execution_view_topology_rebuilds_total,
        )

    def _compose_resident_state(
        self,
        slices: tuple[Any, ...],
    ) -> ResidentStateView:
        slab_token_counts: dict[int, int] = {}
        total_tokens = 0
        for slice_ref in slices:
            total_tokens += slice_ref.token_count
            slab_token_counts[slice_ref.slab_id] = (
                slab_token_counts.get(slice_ref.slab_id, 0) + slice_ref.token_count
            )
        return ResidentStateView(
            total_tokens=total_tokens,
            slices=slices,
            topology_epoch=self._topology_epoch,
            tail_epoch=self._tail_epoch,
            n_execution_slabs=len(slab_token_counts),
            slab_token_counts=tuple(slab_token_counts.values()),
            fabric_compactions_total=self._backend.compactions_total,
            execution_view_topology_rebuilds_total=self._execution_view_topology_rebuilds_total,
            execution_view_local_repairs_total=self._execution_view_local_repairs_total,
            execution_view_repaired_suffix_tokens_total=(
                self._execution_view_repaired_suffix_tokens_total
            ),
        )

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        del window_size
        return _mask_from_length(n, offset=self._logical_offset, return_array=return_array)

    def reset(self) -> None:
        self._logical_offset = 0
        self._handles = {}
        self._resident_state = None
        self._backend.reset()
        self._topology_epoch = 0
        self._tail_epoch = 0
        self._resident_view_topology_epoch = -1
        self._resident_view_tail_epoch = -1
        self._resident_view_query_key = None
        self._resident_view_visible_start = 0
        self._execution_view_topology_rebuilds_total = 0
        self._execution_view_local_repairs_total = 0
        self._execution_view_repaired_suffix_tokens_total = 0
        self._cold_batch_depth = 0
        self._cold_topology_dirty = False
        self._append_only_topology_pending = False
        self._append_only_repair_logical_start = None

    def trim(self, n: int) -> int:
        if n <= 0:
            return 0
        new_offset = max(0, self._logical_offset - n)
        retained: dict[int, ResidentBlockHandle] = {}
        self.begin_cold_mutation_batch()
        try:
            for block_id, handle in self._handles.items():
                if handle.logical_span[0] < new_offset:
                    retained[block_id] = handle
                else:
                    self._backend.evict_handle(handle)
        finally:
            self.end_cold_mutation_batch()
        self._handles = retained
        trimmed = self._logical_offset - new_offset
        self._logical_offset = new_offset
        self._mark_topology_change()
        self._manager.bump_resident_version()
        return trimmed

    def begin_cold_mutation_batch(self) -> None:
        self._cold_batch_depth += 1
        self._backend.begin_cold_mutation_batch()

    def end_cold_mutation_batch(self) -> None:
        if self._cold_batch_depth <= 0:
            raise RuntimeError("Cold mutation batch underflow in full_kv layer cache")
        self._cold_batch_depth -= 1
        self._backend.end_cold_mutation_batch()
        if self._cold_batch_depth == 0 and self._cold_topology_dirty:
            self._topology_epoch += 1
            self._cold_topology_dirty = False
            self._append_only_topology_pending = False
            self._append_only_repair_logical_start = None

    def should_sample_usage(self) -> bool:
        return True

    def required_history_start(self, history_tokens: int) -> int:
        del history_tokens
        return 0

    def record_usage_from_attention(
        self,
        resident_state: ResidentStateView,
        usage_by_token: mx.array,
    ) -> None:
        started_ns = time.perf_counter_ns()
        for slice_ref in resident_state.slices:
            seg_start, seg_end = slice_ref.resident_slice
            segment_usage = usage_by_token[seg_start:seg_end]
            values: list[mx.array] = []
            block_ids: list[int] = []
            for block_id, local_start, local_end in slice_ref.block_slices:
                block_ids.append(block_id)
                values.append(segment_usage[local_start:local_end].mean(keepdims=True))
            if values:
                self._manager.usage.record_batch(
                    tuple(block_ids),
                    values[0] if len(values) == 1 else mx.concatenate(values, axis=0),
                )
        self.record_perf_ns(
            "usage.record_from_attention_ns",
            time.perf_counter_ns() - started_ns,
        )

    def block_live_bytes(self, block_id: int) -> int:
        handle = self._handles.get(block_id)
        return 0 if handle is None else handle.live_bytes

    def _append_to_block(
        self,
        block: BlockRecord,
        keys: mx.array,
        values: mx.array,
    ) -> bool:
        handle = self._handles.get(block.block_id)
        if handle is None:
            self._handles[block.block_id] = self._backend.create_handle(
                block_id=block.block_id,
                profile=block.profile,
                logical_span=(block.start_token, block.start_token + keys.shape[2]),
                keys=keys,
                values=values,
            )
            return False
        converted = False
        if handle.profile is not block.profile:
            handle = self._backend.convert_profile(
                handle,
                profile=block.profile,
                dtype=self._effective_dtype(),
            )
            self._handles[block.block_id] = handle
            converted = True
        old_fragments = handle.fragments
        self._handles[block.block_id] = self._backend.append_tokens(
            handle,
            logical_span=(block.start_token, block.end_token),
            keys=keys,
            values=values,
            dtype=self._effective_dtype(),
        )
        new_handle = self._handles[block.block_id]
        if converted:
            return False
        if len(old_fragments) != len(new_handle.fragments):
            return False
        if not old_fragments:
            return False
        for prev, curr in zip(old_fragments[:-1], new_handle.fragments[:-1], strict=True):
            if prev != curr:
                return False
        return (
            old_fragments[-1].slab_id == new_handle.fragments[-1].slab_id
            and old_fragments[-1].local_start == new_handle.fragments[-1].local_start
            and old_fragments[-1].local_end <= new_handle.fragments[-1].local_end
        )

    def _ordered_handles(self, history_tokens: int) -> list[ResidentBlockHandle]:
        ordered: list[ResidentBlockHandle] = []
        for block in self._manager.registry.covered_resident_blocks(history_tokens):
            handle = self._handles.get(block.block_id)
            if handle is None:
                raise AdaptiveKVError(
                    f"Resident block {block.block_id} is missing a TurboQuant handle"
                )
            ordered.append(handle)
        return ordered

    def _effective_dtype(self) -> mx.Dtype:
        return self._dtype if self._dtype is not None else mx.float32

    def _debug_arrays(self) -> tuple[mx.array | None, mx.array | None]:
        ordered = self._ordered_handles(self._logical_offset)
        if not ordered:
            return None, None
        return self._backend.debug_materialize(tuple(ordered))

    def _mark_topology_change(self) -> None:
        if self._cold_batch_depth > 0:
            self._cold_topology_dirty = True
            self._append_only_topology_pending = False
            self._append_only_repair_logical_start = None
            return
        self._topology_epoch += 1
        self._append_only_topology_pending = False
        self._append_only_repair_logical_start = None

    def record_perf_ns(self, name: str, elapsed_ns: int) -> None:
        self._manager.record_perf_ns(name, elapsed_ns)

    def increment_perf(self, name: str, delta: int = 1) -> None:
        self._manager.increment_perf(name, delta)

    def perf_sync_enabled(self) -> bool:
        return self._manager.perf_sync_enabled()


def make_full_kv_family_bindings() -> RuntimeFamilyBindings:
    return RuntimeFamilyBindings(
        descriptor=FULL_KV_FAMILY,
        runtime_substrate=FullAttentionKVRuntimeSubstrate(),
        replay_backend=FullAttentionKVReplayBackend(),
        layer_runtime_type=FullAttentionKVAdaptiveLayerCache,
    )


__all__ = [
    "FULL_KV_FAMILY",
    "FullAttentionKVAdaptiveLayerCache",
    "FullAttentionKVReplayBackend",
    "FullAttentionKVRuntimeSubstrate",
    "make_full_kv_family_bindings",
]
