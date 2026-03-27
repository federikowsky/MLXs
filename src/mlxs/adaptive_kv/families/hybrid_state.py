"""Family C runtime substrate: hybrid recurrent/stateful + KV decoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx

from mlxs.adaptive_kv.families.full_kv import (
    FullAttentionKVAdaptiveLayerCache,
    FullAttentionKVReplayBackend,
)
from mlxs.adaptive_kv.runtime import (
    AdaptiveKVLayerRuntime,
    AdaptiveKVReplayBackend,
    AdaptiveKVRuntimeSubstrate,
    RuntimeFamily,
    RuntimeFamilyBindings,
    RuntimeFamilyDescriptor,
    ScratchReplayState,
)
from mlxs.cache.arrays import ArraysCache

if TYPE_CHECKING:
    from mlxs.adaptive_kv.manager import AdaptiveKVManager


HYBRID_STATE_FAMILY = RuntimeFamilyDescriptor(
    family=RuntimeFamily.HYBRID_STATE,
    display_name="Hybrid-State Decoder",
    summary=(
        "Mixed recurrent or linear state plus token-addressable KV state across layers, "
        "with Adaptive KV applied to the KV-bearing layers and exact pass-through state "
        "for the recurrent layers."
    ),
    token_addressable=False,
    hybrid_state=True,
)


class HybridStateArraysLayerCache:
    """Adaptive-KV-compatible wrapper for recurrent/state-array layers."""

    __slots__ = ("_cache", "_layer_index", "_logical_offset", "_manager")

    def __init__(self, manager: AdaptiveKVManager, layer_index: int) -> None:
        self._manager = manager
        self._layer_index = layer_index
        self._cache = ArraysCache(size=2)
        self._logical_offset = 0

    def __getitem__(self, idx: int) -> Any:
        return self._cache[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._cache[idx] = value

    @property
    def offset(self) -> int:
        return self._logical_offset

    @property
    def keys(self) -> None:
        return None

    @property
    def values(self) -> None:
        return None

    @property
    def state(self) -> tuple[Any, ...] | None:
        if self._cache.empty():
            return None
        return tuple(self._cache.cache)

    @property
    def live_state_size_bytes(self) -> int:
        return self._cache.state_size_bytes

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[Any, Any]:
        raise RuntimeError(
            "Hybrid-state recurrent layers do not support KV update_and_fetch"
        )

    def remove_token_range(self, start: int, end: int) -> None:
        del start, end

    def demote_block(self, block_id: int) -> None:
        del block_id

    def promote_block(self, block_id: int) -> None:
        del block_id

    def evict_block(self, block_id: int) -> None:
        del block_id

    def recover_block(self, block_id: int, keys: mx.array, values: mx.array) -> None:
        del block_id, keys, values

    def recover_blocks(
        self,
        blocks: tuple[Any, ...],
        keys: mx.array,
        values: mx.array,
    ) -> None:
        del blocks, keys, values

    def recover_blocks_from_scratch(
        self,
        blocks: tuple[Any, ...],
        replay_layer: Any,
        replay_backend: AdaptiveKVReplayBackend,
    ) -> None:
        del blocks, replay_layer, replay_backend

    def resident_state_for_attention(self) -> Any:
        raise RuntimeError(
            "Hybrid-state recurrent layers do not expose resident attention state"
        )

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | None:
        del return_array, window_size
        return self._cache.make_mask(n)

    def reset(self) -> None:
        self._cache.reset()
        self._logical_offset = 0

    def trim(self, n: int) -> int:
        self._logical_offset = max(0, self._logical_offset - n)
        return self._cache.trim(n)

    def should_sample_usage(self) -> bool:
        return False

    def record_usage_from_attention(
        self,
        resident_state: Any,
        usage_by_token: mx.array,
    ) -> None:
        del resident_state, usage_by_token

    def block_live_bytes(self, block_id: int) -> int:
        del block_id
        return 0

    def prepare(self, lengths: list[int] | None = None, **kwargs: Any) -> None:
        self._cache.prepare(lengths=lengths, **kwargs)

    def finalize(self) -> None:
        self._cache.finalize()

    def advance(self, n: int) -> None:
        self._cache.advance(n)
        self._logical_offset += n


class HybridStateKVAdaptiveLayerCache(FullAttentionKVAdaptiveLayerCache):
    """KV-bearing hybrid-state layers keep compressed resident storage but dequantize
    compressed segments during attention execution to preserve token fidelity."""

    def _dequantize_compressed_attention(self) -> bool:
        return True


class HybridStateAdaptiveLayerCache:
    """Lazy per-layer delegate: recurrent layers pass through, KV layers stay adaptive."""

    __slots__ = ("_delegate", "_layer_index", "_manager")

    def __init__(self, manager: AdaptiveKVManager, layer_index: int) -> None:
        self._manager = manager
        self._layer_index = layer_index
        self._delegate: AdaptiveKVLayerRuntime | HybridStateArraysLayerCache | None = None

    def _resolve_delegate(self) -> AdaptiveKVLayerRuntime | HybridStateArraysLayerCache:
        if self._delegate is not None:
            return self._delegate
        model = self._manager._model
        if model is None:
            raise RuntimeError(
                "Hybrid-state Adaptive KV layer runtime requires a bound generation model"
            )
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or not (0 <= self._layer_index < len(layers)):
            raise RuntimeError(
                "Hybrid-state Adaptive KV could not resolve the model layer layout"
            )
        layer = layers[self._layer_index]
        if getattr(layer, "is_linear", False):
            self._delegate = HybridStateArraysLayerCache(
                self._manager,
                layer_index=self._layer_index,
            )
        else:
            self._delegate = HybridStateKVAdaptiveLayerCache(
                self._manager,
                layer_index=self._layer_index,
            )
        return self._delegate

    def __getitem__(self, idx: int) -> Any:
        delegate = self._resolve_delegate()
        return delegate[idx]  # type: ignore[index]

    def __setitem__(self, idx: int, value: Any) -> None:
        delegate = self._resolve_delegate()
        delegate[idx] = value  # type: ignore[index]

    @property
    def offset(self) -> int:
        return self._resolve_delegate().offset

    @property
    def keys(self) -> mx.array | None:
        return self._resolve_delegate().keys

    @property
    def values(self) -> mx.array | None:
        return self._resolve_delegate().values

    @property
    def state(self) -> tuple[Any, ...] | None:
        return self._resolve_delegate().state

    @property
    def live_state_size_bytes(self) -> int:
        return self._resolve_delegate().live_state_size_bytes

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[Any, Any]:
        return self._resolve_delegate().update_and_fetch(keys, values)

    def remove_token_range(self, start: int, end: int) -> None:
        self._resolve_delegate().remove_token_range(start, end)

    def demote_block(self, block_id: int) -> None:
        self._resolve_delegate().demote_block(block_id)

    def promote_block(self, block_id: int) -> None:
        self._resolve_delegate().promote_block(block_id)

    def evict_block(self, block_id: int) -> None:
        self._resolve_delegate().evict_block(block_id)

    def recover_block(self, block_id: int, keys: mx.array, values: mx.array) -> None:
        self._resolve_delegate().recover_block(block_id, keys, values)

    def recover_blocks(
        self,
        blocks: tuple[Any, ...],
        keys: mx.array,
        values: mx.array,
    ) -> None:
        self._resolve_delegate().recover_blocks(blocks, keys, values)

    def recover_blocks_from_scratch(
        self,
        blocks: tuple[Any, ...],
        replay_layer: Any,
        replay_backend: AdaptiveKVReplayBackend,
    ) -> None:
        self._resolve_delegate().recover_blocks_from_scratch(
            blocks,
            replay_layer,
            replay_backend,
        )

    def resident_state_for_attention(self) -> Any:
        return self._resolve_delegate().resident_state_for_attention()

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        return self._resolve_delegate().make_mask(
            n,
            return_array=return_array,
            window_size=window_size,
        )

    def reset(self) -> None:
        self._resolve_delegate().reset()

    def trim(self, n: int) -> int:
        return self._resolve_delegate().trim(n)

    def should_sample_usage(self) -> bool:
        return self._resolve_delegate().should_sample_usage()

    def record_usage_from_attention(
        self,
        resident_state: Any,
        usage_by_token: mx.array,
    ) -> None:
        self._resolve_delegate().record_usage_from_attention(resident_state, usage_by_token)

    def block_live_bytes(self, block_id: int) -> int:
        return self._resolve_delegate().block_live_bytes(block_id)

    def prepare(self, lengths: list[int] | None = None, **kwargs: Any) -> None:
        delegate = self._resolve_delegate()
        prepare = getattr(delegate, "prepare", None)
        if prepare is not None:
            prepare(lengths=lengths, **kwargs)

    def finalize(self) -> None:
        delegate = self._resolve_delegate()
        finalize = getattr(delegate, "finalize", None)
        if finalize is not None:
            finalize()

    def advance(self, n: int) -> None:
        delegate = self._resolve_delegate()
        advance = getattr(delegate, "advance", None)
        if advance is not None:
            advance(n)


class HybridStateRuntimeSubstrate(AdaptiveKVRuntimeSubstrate):
    """Runtime substrate for hybrid recurrent/stateful + KV decoder families."""

    def make_layer_runtime(
        self,
        manager: AdaptiveKVManager,
        layer_index: int,
    ) -> HybridStateAdaptiveLayerCache:
        return HybridStateAdaptiveLayerCache(manager, layer_index=layer_index)


class HybridStateReplayBackend(FullAttentionKVReplayBackend):
    """Replay backend for mixed ArraysCache/KVCache language runtimes."""

    @staticmethod
    def _cache_eval_tensors(cache: Any) -> list[mx.array]:
        state = getattr(cache, "state", None)
        if state is not None:
            if isinstance(state, tuple):
                return [tensor for tensor in state if hasattr(tensor, "nbytes")]
            if hasattr(state, "nbytes"):
                return [state]
        cache_list = getattr(cache, "cache", None)
        if cache_list is None:
            return []
        return [tensor for tensor in cache_list if hasattr(tensor, "nbytes")]

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
        del num_layers
        total_tokens = max(0, min(total_tokens, len(source_tokens)))
        if scratch_cache is None:
            scratch_cache = model.make_cache()
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


def make_hybrid_state_family_bindings() -> RuntimeFamilyBindings:
    return RuntimeFamilyBindings(
        descriptor=HYBRID_STATE_FAMILY,
        runtime_substrate=HybridStateRuntimeSubstrate(),
        replay_backend=HybridStateReplayBackend(),
        layer_runtime_type=HybridStateAdaptiveLayerCache,
    )


__all__ = [
    "HYBRID_STATE_FAMILY",
    "HybridStateAdaptiveLayerCache",
    "HybridStateArraysLayerCache",
    "HybridStateReplayBackend",
    "HybridStateRuntimeSubstrate",
    "make_hybrid_state_family_bindings",
]
