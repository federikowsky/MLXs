"""Adaptive KV manager and cache wrapper."""

from __future__ import annotations

import math
import time
from typing import Any

import mlx.core as mx

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
from mlxs.adaptive_kv.scoring import AdaptiveScoreEngine
from mlxs.adaptive_kv.storage import (
    AdaptiveAttentionSegment,
    AdaptiveCompressedRunStore,
    AdaptiveResidentState,
)
from mlxs.adaptive_kv.transitions import AdaptiveTransitionEngine
from mlxs.adaptive_kv.usage import AdaptiveUsageCollector
from mlxs.cache.attention_mask import _mask_from_length
from mlxs.cache.kv import KVCache


class AdaptiveLayerCache:
    """Per-layer adaptive cache wrapper for the supported llama baseline."""

    def __init__(self, manager: AdaptiveKVManager, layer_index: int) -> None:
        self._manager = manager
        self._layer_index = layer_index
        self._logical_offset = 0
        self._full_cache = KVCache()
        self._full_block_slices: dict[int, tuple[int, int]] = {}
        self._compressed_runs: dict[int, AdaptiveCompressedRunStore] = {}
        self._compressed_block_runs: dict[int, int] = {}
        self._next_compressed_run_id = 0
        self._resident_state: AdaptiveResidentState | None = None
        self._resident_state_version = -1
        self._assembled_keys: mx.array | None = None
        self._assembled_values: mx.array | None = None
        self._assembled_version = -1
        self._dtype: Any = None

    @property
    def offset(self) -> int:
        return self._logical_offset

    @property
    def keys(self) -> mx.array | None:
        keys, _ = self._assemble_debug_arrays()
        return keys

    @property
    def values(self) -> mx.array | None:
        _, values = self._assemble_debug_arrays()
        return values

    @property
    def state(self) -> tuple[Any, ...] | None:
        tensors: list[Any] = []
        if self._full_cache.state is not None:
            tensors.append(self._full_cache.state)
        for run in self._ordered_compressed_runs():
            tensors.append(run.state_tensors())
        return tuple(tensors) if tensors else None

    @property
    def state_size_bytes(self) -> int:
        return self.live_state_size_bytes

    @property
    def live_state_size_bytes(self) -> int:
        total = self._full_cache.live_state_size_bytes
        total += sum(run.live_bytes for run in self._compressed_runs.values())
        return total

    @property
    def resident_token_count(self) -> int:
        return self._full_cache.offset + sum(
            run.token_count for run in self._compressed_runs.values()
        )

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[Any, Any]:
        start = self._logical_offset
        end = start + keys.shape[2]
        if self._dtype is None:
            self._dtype = keys.dtype
        self._manager.ensure_block_coverage(end)
        for block_id, local_start, local_end in self._manager.registry.token_slices(start, end):
            block = self._manager.registry.get(block_id)
            if block.tier is not BlockTier.FULL:
                block = self._manager.force_full_for_append(block_id)
            self._append_to_full_block(
                block.block_id,
                keys[..., local_start:local_end, :],
                values[..., local_start:local_end, :],
            )
        self._logical_offset = end
        self._manager.bump_resident_version()
        resident_state = self.resident_state_for_attention()
        return resident_state, resident_state

    def remove_token_range(self, start: int, end: int) -> None:
        """Best-effort resident-range removal for guarded maintenance paths."""
        if end <= start:
            return
        cursor = 0
        removals: list[int] = []
        for segment in self.resident_state_for_attention().segments:
            segment_start = cursor
            segment_end = cursor + segment.token_count
            cursor = segment_end
            if end <= segment_start or start >= segment_end:
                continue
            for block_id, _, _ in segment.block_slices:
                removals.append(block_id)
        for block_id in removals:
            if block_id in self._compressed_block_runs:
                self._remove_compressed_block(block_id)
            if block_id in self._full_block_slices:
                self._remove_full_block(block_id)
        if removals:
            self._manager.bump_resident_version()

    def demote_block(self, block_id: int) -> None:
        if block_id in self._compressed_block_runs:
            return
        full_keys, full_values = self._full_block_tensors(block_id)
        if full_keys is None or full_values is None:
            raise AdaptiveKVError(f"Cannot demote block {block_id}: missing full resident tensors")
        group_size = self._quantized_group_size(
            k_head_dim=full_keys.shape[-1],
            v_head_dim=full_values.shape[-1],
        )
        run = AdaptiveCompressedRunStore.from_full_block(
            self._allocate_compressed_run_id(),
            block_id=block_id,
            keys=full_keys,
            values=full_values,
            group_size=group_size,
            bits=8,
        )
        self._remove_full_block(block_id)
        self._insert_compressed_run(block_id, run)
        self._manager.bump_resident_version()

    def promote_block(self, block_id: int) -> None:
        run = self._compressed_run(block_id)
        if run is None:
            raise AdaptiveKVError(
                f"Cannot promote block {block_id}: missing compressed resident state"
            )
        keys, values = run.block_tensors(
            block_id,
            dtype=self._dtype if self._dtype is not None else mx.float32,
        )
        self._remove_compressed_block(block_id)
        self._insert_full_block(block_id, keys, values)
        self._manager.bump_resident_version()

    def evict_block(self, block_id: int) -> None:
        if block_id in self._compressed_block_runs:
            self._remove_compressed_block(block_id)
            self._manager.bump_resident_version()
            return
        if block_id in self._full_block_slices:
            raise AdaptiveKVError(
                f"Adaptive hard eviction requires COMPRESSED resident state first; "
                f"block {block_id} is still FULL"
            )

    def recover_block(self, block_id: int, keys: mx.array, values: mx.array) -> None:
        if block_id in self._full_block_slices or block_id in self._compressed_block_runs:
            return
        expected = self._manager.registry.get(block_id).token_count
        if keys.shape[2] != expected or values.shape[2] != expected:
            raise AdaptiveKVError(
                f"Recovered block {block_id} has inconsistent token count: "
                f"expected {expected}, got keys={keys.shape[2]}, values={values.shape[2]}"
            )
        group_size = self._quantized_group_size(
            k_head_dim=keys.shape[-1],
            v_head_dim=values.shape[-1],
        )
        run = AdaptiveCompressedRunStore.from_full_block(
            self._allocate_compressed_run_id(),
            block_id=block_id,
            keys=keys,
            values=values,
            group_size=group_size,
            bits=8,
        )
        self._insert_compressed_run(block_id, run)
        self._manager.bump_resident_version()

    def recover_blocks(
        self,
        blocks: tuple[BlockRecord, ...],
        keys: mx.array,
        values: mx.array,
    ) -> None:
        if not blocks:
            return
        if len(blocks) == 1:
            self.recover_block(blocks[0].block_id, keys, values)
            return
        if any(
            block.block_id in self._full_block_slices
            or block.block_id in self._compressed_block_runs
            for block in blocks
        ):
            return
        expected = sum(block.token_count for block in blocks)
        if keys.shape[2] != expected or values.shape[2] != expected:
            raise AdaptiveKVError(
                "Recovered block run has inconsistent token count: "
                f"expected {expected}, got keys={keys.shape[2]}, values={values.shape[2]}"
            )
        group_size = self._quantized_group_size(
            k_head_dim=keys.shape[-1],
            v_head_dim=values.shape[-1],
        )
        cursor = 0
        block_slices: list[tuple[int, int, int]] = []
        for block in blocks:
            next_cursor = cursor + block.token_count
            block_slices.append((block.block_id, cursor, next_cursor))
            cursor = next_cursor
        run = AdaptiveCompressedRunStore.from_full_run(
            self._allocate_compressed_run_id(),
            block_slices=tuple(block_slices),
            keys=keys,
            values=values,
            group_size=group_size,
            bits=8,
        )
        self._insert_compressed_run_span(
            first_block_id=blocks[0].block_id,
            last_block_id=blocks[-1].block_id,
            run=run,
        )
        self._manager.bump_resident_version()

    def resident_state_for_attention(self) -> AdaptiveResidentState:
        if self._resident_state_version == self._manager.resident_version:
            if self._resident_state is None:
                raise AdaptiveKVError("Adaptive resident state cache is unexpectedly empty")
            return self._resident_state

        segments: list[AdaptiveAttentionSegment] = []
        run_slice: tuple[int, int] | None = None
        run_blocks: list[tuple[int, int, int]] = []

        def flush_full_run() -> None:
            nonlocal run_slice, run_blocks
            if run_slice is None or not run_blocks:
                run_slice = None
                run_blocks = []
                return
            segments.append(
                AdaptiveAttentionSegment(
                    tier=BlockTier.FULL,
                    token_count=run_slice[1] - run_slice[0],
                    block_slices=tuple(run_blocks),
                    full_slice=run_slice,
                )
            )
            run_slice = None
            run_blocks = []

        emitted_compressed_runs: set[int] = set()
        for block in self._manager.registry.covered_resident_blocks(self._logical_offset):
            if block.tier is BlockTier.FULL:
                block_slice = self._full_block_slices.get(block.block_id)
                if block_slice is None:
                    raise AdaptiveKVError(
                        f"Adaptive resident state is missing FULL slice for block {block.block_id}"
                    )
                local_start = 0 if run_slice is None else block_slice[0] - run_slice[0]
                local_end = local_start + (block_slice[1] - block_slice[0])
                if run_slice is None:
                    run_slice = block_slice
                    run_blocks = [(block.block_id, local_start, local_end)]
                elif block_slice[0] == run_slice[1]:
                    run_slice = (run_slice[0], block_slice[1])
                    run_blocks.append((block.block_id, local_start, local_end))
                else:
                    flush_full_run()
                    run_slice = block_slice
                    run_blocks = [(block.block_id, 0, block_slice[1] - block_slice[0])]
                continue

            flush_full_run()
            run_id = self._compressed_block_runs.get(block.block_id)
            if run_id is None:
                raise AdaptiveKVError(
                    f"Adaptive resident state is missing COMPRESSED run for block {block.block_id}"
                )
            if run_id in emitted_compressed_runs:
                continue
            run = self._compressed_runs.get(run_id)
            if run is None:
                raise AdaptiveKVError(
                    f"Adaptive resident state is missing COMPRESSED run store for block "
                    f"{block.block_id}"
                )
            segments.append(
                AdaptiveAttentionSegment(
                    tier=BlockTier.COMPRESSED,
                    token_count=run.token_count,
                    block_slices=run.block_slices,
                    q_keys=run.q_keys,
                    q_values=run.q_values,
                    group_size=run.group_size,
                    bits=run.bits,
                )
            )
            emitted_compressed_runs.add(run_id)

        flush_full_run()
        resident_token_count = sum(segment.token_count for segment in segments)
        if resident_token_count != self._logical_offset:
            raise AdaptiveKVError(
                "Adaptive resident state is incomplete for attention: "
                f"resident_tokens={resident_token_count}, logical_offset={self._logical_offset}"
            )
        self._resident_state = AdaptiveResidentState(
            total_tokens=self._logical_offset,
            segments=tuple(segments),
            full_keys=self._full_cache.keys,
            full_values=self._full_cache.values,
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
        return _mask_from_length(
            n,
            offset=self._logical_offset,
            return_array=return_array,
            window_size=window_size,
        )

    def reset(self) -> None:
        self._full_cache.reset()
        self._full_block_slices = {}
        self._compressed_runs = {}
        self._compressed_block_runs = {}
        self._next_compressed_run_id = 0
        self._logical_offset = 0
        self._resident_state = None
        self._resident_state_version = -1
        self._assembled_keys = None
        self._assembled_values = None
        self._assembled_version = -1
        self._dtype = None

    def trim(self, n: int) -> int:
        del n
        return 0

    def should_sample_usage(self) -> bool:
        return self._manager.should_sample_usage()

    def record_usage_from_attention(
        self,
        resident_state: AdaptiveResidentState,
        usage_by_token: mx.array,
    ) -> None:
        cursor = 0
        block_ids: list[int] = []
        block_usage: list[mx.array] = []
        for segment in resident_state.segments:
            segment_usage = usage_by_token[cursor : cursor + segment.token_count]
            for block_id, local_start, local_end in segment.block_slices:
                block_ids.append(block_id)
                block_usage.append(segment_usage[local_start:local_end].sum())
            cursor += segment.token_count
        if not block_usage:
            return
        usage_values = mx.stack(block_usage, axis=0)
        mx.async_eval(usage_values)
        self._manager.usage.record_batch(tuple(block_ids), usage_values)

    def block_live_bytes(self, block_id: int) -> int:
        run = self._compressed_run(block_id)
        if run is not None:
            return run.block_live_bytes(block_id)
        block_slice = self._full_block_slices.get(block_id)
        if block_slice is None:
            return 0
        keys = self._full_cache.keys
        values = self._full_cache.values
        if keys is None or values is None:
            return 0
        start, end = block_slice
        return keys[..., start:end, :].nbytes + values[..., start:end, :].nbytes

    def _append_to_full_block(
        self,
        block_id: int,
        keys: mx.array,
        values: mx.array,
    ) -> None:
        block_slice = self._full_block_slices.get(block_id)
        if block_slice is None:
            if block_id in self._compressed_block_runs:
                raise AdaptiveKVError(
                    f"Cannot append into COMPRESSED block {block_id} without promotion"
                )
            start = self._full_cache.offset
            self._full_cache.update_and_fetch(keys, values)
            self._full_block_slices[block_id] = (start, self._full_cache.offset)
            return

        if block_slice[1] != self._full_cache.offset:
            raise AdaptiveKVError(
                f"Adaptive append block {block_id} must be the trailing FULL resident slice"
            )
        self._full_cache.update_and_fetch(keys, values)
        self._full_block_slices[block_id] = (block_slice[0], self._full_cache.offset)

    def _full_block_tensors(self, block_id: int) -> tuple[mx.array | None, mx.array | None]:
        block_slice = self._full_block_slices.get(block_id)
        keys = self._full_cache.keys
        values = self._full_cache.values
        if block_slice is None or keys is None or values is None:
            return None, None
        start, end = block_slice
        return keys[..., start:end, :], values[..., start:end, :]

    def _insert_full_block(
        self,
        block_id: int,
        keys: mx.array,
        values: mx.array,
    ) -> None:
        insert_at = self._full_insert_position(block_id)
        current_keys = self._full_cache.keys
        current_values = self._full_cache.values
        if current_keys is None or current_values is None:
            self._full_cache.state = (keys, values)
            self._full_block_slices[block_id] = (0, keys.shape[2])
            return

        pieces_k: list[mx.array] = []
        pieces_v: list[mx.array] = []
        if insert_at > 0:
            pieces_k.append(current_keys[..., :insert_at, :])
            pieces_v.append(current_values[..., :insert_at, :])
        pieces_k.append(keys)
        pieces_v.append(values)
        if insert_at < current_keys.shape[2]:
            pieces_k.append(current_keys[..., insert_at:, :])
            pieces_v.append(current_values[..., insert_at:, :])
        new_keys = pieces_k[0] if len(pieces_k) == 1 else mx.concatenate(pieces_k, axis=2)
        new_values = pieces_v[0] if len(pieces_v) == 1 else mx.concatenate(pieces_v, axis=2)
        self._full_cache.state = (new_keys, new_values)
        self._shift_full_slices(insert_at, keys.shape[2])
        self._full_block_slices[block_id] = (insert_at, insert_at + keys.shape[2])

    def _remove_full_block(self, block_id: int) -> None:
        block_slice = self._full_block_slices.pop(block_id, None)
        if block_slice is None:
            return
        start, end = block_slice
        removed = self._full_cache.remove_token_range(start, end)
        if removed == 0:
            return
        self._shift_full_slices(end, -removed)

    def _shift_full_slices(self, start: int, delta: int) -> None:
        if delta == 0:
            return
        for block_id, (slice_start, slice_end) in list(self._full_block_slices.items()):
            if slice_start >= start:
                self._full_block_slices[block_id] = (
                    slice_start + delta,
                    slice_end + delta,
                )

    def _full_insert_position(self, block_id: int) -> int:
        cursor = 0
        for block in self._manager.registry.snapshot():
            if block.block_id == block_id:
                return cursor
            full_slice = self._full_block_slices.get(block.block_id)
            if full_slice is not None:
                cursor = full_slice[1]
        return cursor

    def _assemble_debug_arrays(self) -> tuple[mx.array | None, mx.array | None]:
        if self._assembled_version == self._manager.resident_version:
            return self._assembled_keys, self._assembled_values
        resident_state = self.resident_state_for_attention()
        if not resident_state.segments:
            self._assembled_keys = None
            self._assembled_values = None
            self._assembled_version = self._manager.resident_version
            return self._assembled_keys, self._assembled_values

        if not resident_state.has_compressed and resident_state.full_keys is not None:
            self._assembled_keys = resident_state.full_keys
            self._assembled_values = resident_state.full_values
            self._assembled_version = self._manager.resident_version
            return self._assembled_keys, self._assembled_values

        keys_parts: list[mx.array] = []
        values_parts: list[mx.array] = []
        full_keys = resident_state.full_keys
        full_values = resident_state.full_values
        for segment in resident_state.segments:
            if segment.tier is BlockTier.FULL:
                if full_keys is None or full_values is None or segment.full_slice is None:
                    continue
                start, end = segment.full_slice
                keys_parts.append(full_keys[..., start:end, :])
                values_parts.append(full_values[..., start:end, :])
                continue
            if segment.q_keys is None or segment.q_values is None:
                continue
            keys = mx.dequantize(
                segment.q_keys[0],
                segment.q_keys[1],
                segment.q_keys[2],
                group_size=segment.group_size,
                bits=segment.bits,
                dtype=self._dtype,
            )
            values = mx.dequantize(
                segment.q_values[0],
                segment.q_values[1],
                segment.q_values[2],
                group_size=segment.group_size,
                bits=segment.bits,
                dtype=self._dtype,
            )
            keys_parts.append(keys)
            values_parts.append(values)

        if not keys_parts:
            self._assembled_keys = None
            self._assembled_values = None
        elif len(keys_parts) == 1:
            self._assembled_keys = keys_parts[0]
            self._assembled_values = values_parts[0]
        else:
            self._assembled_keys = mx.concatenate(keys_parts, axis=2)
            self._assembled_values = mx.concatenate(values_parts, axis=2)
        self._assembled_version = self._manager.resident_version
        return self._assembled_keys, self._assembled_values

    def _allocate_compressed_run_id(self) -> int:
        run_id = self._next_compressed_run_id
        self._next_compressed_run_id += 1
        return run_id

    def _ordered_compressed_runs(self) -> list[AdaptiveCompressedRunStore]:
        ordered: list[AdaptiveCompressedRunStore] = []
        seen: set[int] = set()
        for block in self._manager.registry.snapshot():
            run_id = self._compressed_block_runs.get(block.block_id)
            if run_id is None or run_id in seen:
                continue
            run = self._compressed_runs.get(run_id)
            if run is None:
                continue
            ordered.append(run)
            seen.add(run_id)
        return ordered

    def _compressed_run(self, block_id: int | None) -> AdaptiveCompressedRunStore | None:
        if block_id is None:
            return None
        run_id = self._compressed_block_runs.get(block_id)
        if run_id is None:
            return None
        return self._compressed_runs.get(run_id)

    def _register_compressed_run(self, run: AdaptiveCompressedRunStore) -> None:
        self._compressed_runs[run.run_id] = run
        for block_id, _, _ in run.block_slices:
            self._compressed_block_runs[block_id] = run.run_id

    def _unregister_compressed_run(self, run: AdaptiveCompressedRunStore) -> None:
        self._compressed_runs.pop(run.run_id, None)
        for block_id, _, _ in run.block_slices:
            if self._compressed_block_runs.get(block_id) == run.run_id:
                del self._compressed_block_runs[block_id]

    def _neighbor_block_ids(self, block_id: int) -> tuple[int | None, int | None]:
        blocks = self._manager.registry.snapshot()
        for idx, block in enumerate(blocks):
            if block.block_id != block_id:
                continue
            prev_id = blocks[idx - 1].block_id if idx > 0 else None
            next_id = blocks[idx + 1].block_id if idx + 1 < len(blocks) else None
            return prev_id, next_id
        return None, None

    def _neighbor_block_ids_for_span(
        self,
        first_block_id: int,
        last_block_id: int,
    ) -> tuple[int | None, int | None]:
        blocks = self._manager.registry.snapshot()
        first_idx = -1
        last_idx = -1
        for idx, block in enumerate(blocks):
            if block.block_id == first_block_id:
                first_idx = idx
            if block.block_id == last_block_id:
                last_idx = idx
        if first_idx < 0 or last_idx < 0 or last_idx < first_idx:
            return None, None
        prev_id = blocks[first_idx - 1].block_id if first_idx > 0 else None
        next_id = blocks[last_idx + 1].block_id if last_idx + 1 < len(blocks) else None
        return prev_id, next_id

    def _compressed_neighbor_runs(
        self,
        block_id: int,
    ) -> tuple[AdaptiveCompressedRunStore | None, AdaptiveCompressedRunStore | None]:
        prev_id, next_id = self._neighbor_block_ids(block_id)
        return self._compressed_run(prev_id), self._compressed_run(next_id)

    def _compressed_neighbor_runs_for_span(
        self,
        *,
        first_block_id: int,
        last_block_id: int,
    ) -> tuple[AdaptiveCompressedRunStore | None, AdaptiveCompressedRunStore | None]:
        prev_id, next_id = self._neighbor_block_ids_for_span(first_block_id, last_block_id)
        return self._compressed_run(prev_id), self._compressed_run(next_id)

    @staticmethod
    def _can_merge_runs(
        left: AdaptiveCompressedRunStore,
        right: AdaptiveCompressedRunStore,
    ) -> bool:
        return (
            left.group_size == right.group_size
            and left.bits == right.bits
            and left.q_keys[0].shape[-1] == right.q_keys[0].shape[-1]
            and left.q_values[0].shape[-1] == right.q_values[0].shape[-1]
        )

    def _insert_compressed_run(
        self,
        block_id: int,
        run: AdaptiveCompressedRunStore,
    ) -> None:
        prev_run, next_run = self._compressed_neighbor_runs(block_id)
        merged = run
        if prev_run is not None and self._can_merge_runs(prev_run, merged):
            merged = prev_run.merge_with(merged, run_id=prev_run.run_id)
            self._unregister_compressed_run(prev_run)
        if (
            next_run is not None
            and next_run.run_id != merged.run_id
            and self._can_merge_runs(merged, next_run)
        ):
            merged = merged.merge_with(next_run, run_id=merged.run_id)
            self._unregister_compressed_run(next_run)
        self._register_compressed_run(merged)

    def _insert_compressed_run_span(
        self,
        *,
        first_block_id: int,
        last_block_id: int,
        run: AdaptiveCompressedRunStore,
    ) -> None:
        prev_run, next_run = self._compressed_neighbor_runs_for_span(
            first_block_id=first_block_id,
            last_block_id=last_block_id,
        )
        merged = run
        if prev_run is not None and self._can_merge_runs(prev_run, merged):
            merged = prev_run.merge_with(merged, run_id=prev_run.run_id)
            self._unregister_compressed_run(prev_run)
        if (
            next_run is not None
            and next_run.run_id != merged.run_id
            and self._can_merge_runs(merged, next_run)
        ):
            merged = merged.merge_with(next_run, run_id=merged.run_id)
            self._unregister_compressed_run(next_run)
        self._register_compressed_run(merged)

    def _remove_compressed_block(self, block_id: int) -> None:
        run = self._compressed_run(block_id)
        if run is None:
            return
        block_start, block_end = run.block_slice(block_id)
        left_run_id = self._allocate_compressed_run_id() if block_start > 0 else None
        right_run_id = self._allocate_compressed_run_id() if block_end < run.token_count else None
        _, fragments = run.split_without_block(
            block_id,
            left_run_id=left_run_id,
            right_run_id=right_run_id,
        )
        self._unregister_compressed_run(run)
        for fragment in fragments:
            self._register_compressed_run(fragment)

    @staticmethod
    def _quantized_group_size(*, k_head_dim: int, v_head_dim: int) -> int:
        common = math.gcd(k_head_dim, v_head_dim)
        for candidate in (128, 64, 32):
            if common % candidate == 0:
                return candidate
        raise AdaptiveKVUnsupportedError(
            "adaptive_kv_v1 compressed tier requires head dimensions divisible by "
            "one of MLX quantization group sizes {32, 64, 128}"
        )


class AdaptiveKVManager:
    """Top-level coordinator for adaptive KV V1."""

    def __init__(self, config: AdaptiveKVConfig, *, num_layers: int, metrics: Any) -> None:
        self.config = config
        self.metrics = metrics
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
        self._layer_caches = [AdaptiveLayerCache(self, layer_index=i) for i in range(num_layers)]
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
        self._scratch_replay_cache: list[KVCache] | None = None
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

    def caches(self) -> list[AdaptiveLayerCache]:
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
        """Diagnostic only: segment structure for mixed-tier attention (layer 0).

        Counts match ``AdaptiveLayerCache.resident_state_for_attention`` — the hot-path
        segment list (FULL runs coalesced; each compressed *run* is one segment).
        """
        if not self._layer_caches:
            return {}
        try:
            rs = self._layer_caches[0].resident_state_for_attention()
        except AdaptiveKVError as exc:
            return {"error": str(exc)}
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
            start_token = recovery_group[0].start_token
            end_token = recovery_group[-1].end_token
            for layer_cache, scratch_layer in zip(self._layer_caches, scratch, strict=True):
                keys, values = scratch_layer.copy_token_range(start_token, end_token)
                expected_tokens = end_token - start_token
                if (
                    keys.shape[2] != expected_tokens
                    or values.shape[2] != expected_tokens
                ):
                    raise AdaptiveKVError(
                        "Replay recovery returned inconsistent run token count: "
                        f"expected {expected_tokens}, got "
                        f"keys={keys.shape[2]}, values={values.shape[2]}"
                    )
                layer_cache.recover_blocks(recovery_group, keys, values)
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

    def _ensure_scratch_replay_prefix(self, total_tokens: int) -> list[KVCache]:
        total_tokens = max(0, min(total_tokens, len(self.source_tokens)))
        if self._scratch_replay_cache is None:
            self._scratch_replay_cache = [KVCache() for _ in range(self._num_layers)]
            self._scratch_replayed_tokens = 0
            self._scratch_replay_materialized = True
        scratch = self._scratch_replay_cache
        offset = self._scratch_replayed_tokens
        if total_tokens <= offset:
            return scratch
        replay_started = time.perf_counter()
        if not self._scratch_replay_materialized and offset > 0:
            mx.eval([cache.state for cache in scratch if cache.state is not None])
            self._scratch_replay_materialized = True
        while offset < total_tokens:
            n = min(self._prefill_step_size, total_tokens - offset)
            chunk = mx.array(self.source_tokens[offset : offset + n])
            self._model(chunk[None], cache=scratch)
            offset += n
            if offset < total_tokens:
                mx.eval([cache.state for cache in scratch if cache.state is not None])
                self._scratch_replay_materialized = True
                mx.clear_cache()
            else:
                self._scratch_replay_materialized = False
        self._scratch_replayed_tokens = total_tokens
        self.metrics.counter(
            REPLAY_FORWARD_TIME_SECONDS_TOTAL,
            time.perf_counter() - replay_started,
        )
        self.metrics.counter(REPLAY_FORWARD_EVENTS_TOTAL)
        return scratch

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
