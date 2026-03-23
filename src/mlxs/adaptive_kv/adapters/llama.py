"""Retained llama-specific Adaptive KV runtime adapter."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from mlxs.adaptive_kv.block_types import BlockRecord, BlockTier
from mlxs.adaptive_kv.exceptions import AdaptiveKVError, AdaptiveKVUnsupportedError
from mlxs.adaptive_kv.runtime import (
    AdapterCapabilities,
    CapabilityStatus,
    ScratchReplayState,
)
from mlxs.adaptive_kv.storage import (
    AdaptiveAttentionSegment,
    AdaptiveCompressedRunStore,
    AdaptiveResidentState,
)
from mlxs.cache.attention_mask import _mask_from_length
from mlxs.cache.kv import KVCache

if TYPE_CHECKING:
    from mlxs.adaptive_kv.manager import AdaptiveKVManager


def _capabilities(
    *,
    adapter_name: str,
    overall: CapabilityStatus,
    num_layers: int = 0,
) -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_name=adapter_name,
        overall=overall,
        baseline_cache=overall,
        resident_attention=overall,
        compressed_tier=overall,
        replay_recovery=overall,
        num_layers=num_layers,
    )


class LlamaAdaptiveKVAdapter:
    """Concrete adapter for the retained full-attention llama baseline."""

    name = "llama"

    def matches_model(self, model: Any) -> bool:
        return getattr(model, "model_type", None) == "llama"

    def assess_generation_support(
        self,
        model: Any,
        *,
        cache: list[Any] | None,
        compile_decode: bool,
        quantized_kv_start: int,
        input_embeddings_present: bool = False,
    ) -> AdapterCapabilities:
        if not self.matches_model(model):
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.unsupported(
                    "adaptive_kv_v1 supports model_type='llama' only"
                ),
            )
        if compile_decode:
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 does not support compile_decode=True"
                ),
            )
        if quantized_kv_start > 0:
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 does not support legacy quantized_kv_start flow"
                ),
            )
        if cache is not None:
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 does not support external cache reuse"
                ),
            )
        if input_embeddings_present:
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 does not support multimodal/input_embeddings requests"
                ),
            )
        if not callable(getattr(model, "make_cache", None)):
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 requires model.make_cache() for the supported llama baseline"
                ),
            )

        args = getattr(model, "args", None)
        layer_types = getattr(args, "layer_types", None)
        if layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 supports the standard full-attention llama baseline only; "
                    "sliding or mixed layer_types are unsupported"
                ),
            )

        model_impl = getattr(model, "model", None)
        layers = getattr(model_impl, "layers", None)
        if layers is not None and any(getattr(layer, "use_sliding", False) for layer in layers):
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 supports the standard full-attention llama baseline only; "
                    "sliding-window llama layers are unsupported"
                ),
            )

        if args is not None:
            hidden_size = getattr(args, "hidden_size", None)
            num_attention_heads = getattr(args, "num_attention_heads", None)
            head_dim = getattr(args, "head_dim", None)
            if head_dim is None and hidden_size is not None and num_attention_heads:
                head_dim = hidden_size // num_attention_heads
            if head_dim is not None and head_dim % 32 != 0:
                return _capabilities(
                    adapter_name=self.name,
                    overall=CapabilityStatus.partial(
                        "adaptive_kv_v1 compressed tier requires llama head_dim divisible by 32"
                    ),
                )

        baseline = model.make_cache()
        if not isinstance(baseline, list) or not baseline:
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 requires a non-empty per-layer cache list"
                ),
            )
        if any(type(layer) is not KVCache for layer in baseline):
            return _capabilities(
                adapter_name=self.name,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 requires a homogeneous list[KVCache] baseline"
                ),
            )
        return _capabilities(
            adapter_name=self.name,
            overall=CapabilityStatus.full(),
            num_layers=len(baseline),
        )

    def make_layer_runtime(
        self,
        manager: AdaptiveKVManager,
        layer_index: int,
    ) -> LlamaAdaptiveLayerCache:
        return LlamaAdaptiveLayerCache(manager, layer_index=layer_index)

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
            mx.eval([cache.state for cache in scratch_cache if cache.state is not None])
            materialized = True
        while offset < total_tokens:
            n = min(prefill_step_size, total_tokens - offset)
            chunk = mx.array(source_tokens[offset : offset + n])
            model(chunk[None], cache=scratch_cache)
            offset += n
            if offset < total_tokens:
                mx.eval([cache.state for cache in scratch_cache if cache.state is not None])
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


class LlamaAdaptiveLayerCache:
    """Per-layer adaptive cache wrapper for the retained llama runtime."""

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
        full_segments: list[AdaptiveAttentionSegment] = []
        compressed_segments: list[AdaptiveAttentionSegment] = []
        run_slice: tuple[int, int] | None = None
        run_blocks: list[tuple[int, int, int]] = []
        resident_cursor = 0

        def append_segment(segment: AdaptiveAttentionSegment) -> None:
            nonlocal resident_cursor
            segments.append(segment)
            if segment.tier is BlockTier.FULL:
                full_segments.append(segment)
            else:
                compressed_segments.append(segment)
            resident_cursor = segment.resident_slice[1]

        def flush_full_run() -> None:
            nonlocal run_slice, run_blocks
            if run_slice is None or not run_blocks:
                run_slice = None
                run_blocks = []
                return
            append_segment(
                AdaptiveAttentionSegment(
                    tier=BlockTier.FULL,
                    token_count=run_slice[1] - run_slice[0],
                    block_slices=tuple(run_blocks),
                    resident_slice=(
                        resident_cursor,
                        resident_cursor + (run_slice[1] - run_slice[0]),
                    ),
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
            append_segment(
                AdaptiveAttentionSegment(
                    tier=BlockTier.COMPRESSED,
                    token_count=run.token_count,
                    block_slices=run.block_slices,
                    resident_slice=(resident_cursor, resident_cursor + run.token_count),
                    q_keys=run.q_keys,
                    q_values=run.q_values,
                    group_size=run.group_size,
                    bits=run.bits,
                )
            )
            emitted_compressed_runs.add(run_id)

        flush_full_run()
        resident_token_count = resident_cursor
        if resident_token_count != self._logical_offset:
            raise AdaptiveKVError(
                "Adaptive resident state is incomplete for attention: "
                f"resident_tokens={resident_token_count}, logical_offset={self._logical_offset}"
            )
        self._resident_state = AdaptiveResidentState(
            total_tokens=self._logical_offset,
            segments=tuple(segments),
            full_segments=tuple(full_segments),
            compressed_segments=tuple(compressed_segments),
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
        block_ids: list[int] = []
        block_usage: list[mx.array] = []
        for segment in resident_state.segments:
            start, end = segment.resident_slice
            segment_usage = usage_by_token[start:end]
            for block_id, local_start, local_end in segment.block_slices:
                block_ids.append(block_id)
                block_usage.append(segment_usage[local_start:local_end].sum())
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
