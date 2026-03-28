"""Family A runtime substrate: TurboQuant-first full-history KV decoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx

from mlxs.adaptive_kv.block_types import BlockRecord, ResidentProfile
from mlxs.adaptive_kv.exceptions import AdaptiveKVError
from mlxs.adaptive_kv.resident import (
    ResidentBlockHandle,
    ResidentEncodedState,
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
from mlxs.adaptive_kv.storage import ResidentAttentionSegment, ResidentStateView
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


def _concat_quantized_states(states: tuple[ResidentEncodedState, ...]) -> ResidentEncodedState:
    if len(states) == 1:
        return states[0]
    parts = list(zip(*states, strict=True))
    return tuple(  # type: ignore[return-value]
        mx.concatenate(list(component), axis=-2) for component in parts
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
        self._resident_state_version = -1
        self._dtype: Any = None

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
        for handle in ordered:
            tensors.extend((handle.q_keys, handle.q_values))
        return tuple(tensors)

    @property
    def live_state_size_bytes(self) -> int:
        return sum(handle.live_bytes for handle in self._handles.values())

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[Any, Any]:
        start = self._logical_offset
        end = start + keys.shape[2]
        if self._dtype is None:
            self._dtype = keys.dtype
        self._manager.ensure_block_coverage(end)
        for block_id, local_start, local_end in self._manager.registry.token_slices(start, end):
            block = self._manager.registry.get(block_id)
            if block.profile is ResidentProfile.EVICTED:
                raise AdaptiveKVError(
                    f"Cannot append into evicted block {block.block_id} without recovery"
                )
            self._append_to_block(
                block,
                keys[..., local_start:local_end, :],
                values[..., local_start:local_end, :],
            )
        self._logical_offset = end
        self._manager.bump_resident_version()
        resident_state = self.resident_state_for_execution(query_tokens=keys.shape[2])
        return resident_state, resident_state

    def remove_token_range(self, start: int, end: int) -> None:
        if end <= start:
            return
        removals = [
            block.block_id
            for block in self._manager.registry.snapshot()
            if block.resident and not (block.end_token <= start or block.start_token >= end)
        ]
        for block_id in removals:
            self._handles.pop(block_id, None)
        if removals:
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
        self._manager.bump_resident_version()

    def evict_block(self, block_id: int) -> None:
        self._handles.pop(block_id, None)
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
        self._manager.bump_resident_version()

    def resident_state_for_execution(self, *, query_tokens: int = 1) -> ResidentStateView:
        del query_tokens
        if self._resident_state_version == self._manager.resident_version and self._resident_state:
            return self._resident_state

        ordered = self._ordered_handles(self._logical_offset)
        segments: list[ResidentAttentionSegment] = []
        resident_cursor = 0
        for handles in self._bounded_execution_groups(ordered):
            first = handles[0]
            token_count = sum(handle.token_count for handle in handles)
            block_slices: list[tuple[int, int, int]] = []
            block_cursor = 0
            for handle in handles:
                next_cursor = block_cursor + handle.token_count
                block_slices.append((handle.block_id, block_cursor, next_cursor))
                block_cursor = next_cursor
            segments.append(
                ResidentAttentionSegment(
                    profile=first.profile,
                    token_count=token_count,
                    logical_span=(handles[0].logical_span[0], handles[-1].logical_span[1]),
                    visible_span=(handles[0].logical_span[0], handles[-1].logical_span[1]),
                    block_slices=tuple(block_slices),
                    resident_slice=(resident_cursor, resident_cursor + token_count),
                    q_keys=_concat_quantized_states(tuple(handle.q_keys for handle in handles)),
                    q_values=_concat_quantized_states(
                        tuple(handle.q_values for handle in handles)
                    ),
                    group_size=first.group_size,
                    bits=first.bits,
                    storage_kind=first.storage_kind,
                    execution_mode=first.execution_mode,
                )
            )
            resident_cursor += token_count

        self._resident_state = ResidentStateView(
            total_tokens=resident_cursor,
            segments=tuple(segments),
        )
        self._resident_state_version = self._manager.resident_version
        return self._resident_state

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
        self._resident_state_version = -1

    def trim(self, n: int) -> int:
        if n <= 0:
            return 0
        new_offset = max(0, self._logical_offset - n)
        self._handles = {
            block_id: handle
            for block_id, handle in self._handles.items()
            if handle.logical_span[0] < new_offset
        }
        trimmed = self._logical_offset - new_offset
        self._logical_offset = new_offset
        self._manager.bump_resident_version()
        return trimmed

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
        for segment in resident_state.segments:
            seg_start, seg_end = segment.resident_slice
            segment_usage = usage_by_token[seg_start:seg_end]
            values: list[mx.array] = []
            block_ids: list[int] = []
            for block_id, local_start, local_end in segment.block_slices:
                block_ids.append(block_id)
                values.append(segment_usage[local_start:local_end].mean(keepdims=True))
            if values:
                self._manager.usage.record_batch(
                    tuple(block_ids),
                    values[0] if len(values) == 1 else mx.concatenate(values, axis=0),
                )

    def block_live_bytes(self, block_id: int) -> int:
        handle = self._handles.get(block_id)
        return 0 if handle is None else handle.live_bytes

    def _append_to_block(
        self,
        block: BlockRecord,
        keys: mx.array,
        values: mx.array,
    ) -> None:
        handle = self._handles.get(block.block_id)
        if handle is None:
            self._handles[block.block_id] = self._backend.create_handle(
                block_id=block.block_id,
                profile=block.profile,
                logical_span=(block.start_token, block.start_token + keys.shape[2]),
                keys=keys,
                values=values,
            )
            return
        if handle.profile is not block.profile:
            handle = self._backend.convert_profile(
                handle,
                profile=block.profile,
                dtype=self._effective_dtype(),
            )
        self._handles[block.block_id] = self._backend.append_tokens(
            handle,
            logical_span=(block.start_token, block.end_token),
            keys=keys,
            values=values,
            dtype=self._effective_dtype(),
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

    def _bounded_execution_groups(
        self,
        ordered: list[ResidentBlockHandle],
    ) -> tuple[tuple[ResidentBlockHandle, ...], ...]:
        if not ordered:
            return ()
        groups: list[list[ResidentBlockHandle]] = [[ordered[0]]]
        current_tokens = ordered[0].token_count
        for handle in ordered[1:]:
            prev = groups[-1][-1]
            contiguous = prev.logical_span[1] == handle.logical_span[0]
            compatible = (
                prev.profile is handle.profile
                and prev.bits == handle.bits
                and prev.group_size == handle.group_size
                and prev.execution_mode is handle.execution_mode
                and prev.storage_kind is handle.storage_kind
                and prev.execution_chunk_tokens == handle.execution_chunk_tokens
                and contiguous
            )
            if compatible and current_tokens + handle.token_count <= prev.execution_chunk_tokens:
                groups[-1].append(handle)
                current_tokens += handle.token_count
            else:
                groups.append([handle])
                current_tokens = handle.token_count
        return tuple(tuple(group) for group in groups)

    def _effective_dtype(self) -> mx.Dtype:
        return self._dtype if self._dtype is not None else mx.float32

    def _debug_arrays(self) -> tuple[mx.array | None, mx.array | None]:
        ordered = self._ordered_handles(self._logical_offset)
        if not ordered:
            return None, None
        keys_list: list[mx.array] = []
        values_list: list[mx.array] = []
        for handle in ordered:
            keys, values = self._backend.materialize(handle)
            keys_list.append(keys)
            values_list.append(values)
        keys = keys_list[0] if len(keys_list) == 1 else mx.concatenate(keys_list, axis=2)
        values = (
            values_list[0]
            if len(values_list) == 1
            else mx.concatenate(values_list, axis=2)
        )
        return keys, values


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
