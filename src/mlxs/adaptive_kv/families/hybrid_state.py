"""Family C runtime substrate: hybrid recurrent/stateful + KV decoders."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from mlxs.adaptive_kv.block_types import ResidentProfile
from mlxs.adaptive_kv.families.full_kv import (
    FullAttentionKVAdaptiveLayerCache,
    FullAttentionKVReplayBackend,
)
from mlxs.adaptive_kv.resident import ResidentExecutionMode, TurboQuantResidentBackend
from mlxs.adaptive_kv.runtime import (
    AdaptiveKVLayerRuntime,
    AdaptiveKVReplayBackend,
    AdaptiveKVRuntimeSubstrate,
    RuntimeFamily,
    RuntimeFamilyBindings,
    RuntimeFamilyDescriptor,
)
from mlxs.cache.arrays import ArraysCache

if TYPE_CHECKING:
    from mlxs.adaptive_kv.manager import AdaptiveKVManager


HYBRID_STATE_FAMILY = RuntimeFamilyDescriptor(
    family=RuntimeFamily.HYBRID_STATE,
    display_name="TurboQuant Hybrid-State Decoder",
    summary=(
        "Mixed recurrent/state-array layers plus TurboQuant-profiled KV layers, with "
        "resident planning applied only to KV-bearing layers."
    ),
    token_addressable=False,
    hybrid_state=True,
)


class HybridStateArraysLayerCache:
    """Pass-through wrapper for recurrent/state-array layers outside profile planning."""

    __slots__ = ("_cache", "_logical_offset")

    def __init__(self) -> None:
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
        raise RuntimeError("Hybrid recurrent layers do not support KV update_and_fetch")

    def remove_token_range(self, start: int, end: int) -> None:
        del start, end

    def degrade_block(self, block_id: int) -> None:
        del block_id

    def restore_block(self, block_id: int) -> None:
        del block_id

    def evict_block(self, block_id: int) -> None:
        del block_id

    def recover_blocks_from_scratch(
        self,
        blocks: tuple[Any, ...],
        replay_layer: Any,
        replay_backend: AdaptiveKVReplayBackend,
        *,
        recovery_profile: ResidentProfile,
    ) -> None:
        del blocks, replay_layer, replay_backend, recovery_profile

    def resident_state_for_execution(self, *, query_tokens: int = 1) -> Any:
        del query_tokens
        raise RuntimeError("Hybrid recurrent layers do not expose resident execution state")

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

    def required_history_start(self, history_tokens: int) -> int:
        return history_tokens

    def record_usage_from_attention(
        self,
        resident_state: Any,
        usage_by_token: mx.array,
    ) -> None:
        del resident_state, usage_by_token

    def block_live_bytes(self, block_id: int) -> int:
        del block_id
        return 0

    def begin_cold_mutation_batch(self) -> None:
        return None

    def end_cold_mutation_batch(self) -> None:
        return None

    def prepare(self, lengths: list[int] | None = None, **kwargs: Any) -> None:
        self._cache.prepare(lengths=lengths, **kwargs)

    def finalize(self) -> None:
        self._cache.finalize()

    def advance(self, n: int) -> None:
        self._cache.advance(n)
        self._logical_offset += n


class HybridStateKVAdaptiveLayerCache(FullAttentionKVAdaptiveLayerCache):
    """KV-bearing Family C layers remain inside resident-profile planning."""


class HybridStateAdaptiveLayerCache:
    """Lazy per-layer delegate with strict KV-vs-state-array planning separation."""

    __slots__ = ("_delegate", "_layer_index", "_manager")

    def __init__(self, manager: AdaptiveKVManager, layer_index: int) -> None:
        self._manager = manager
        self._layer_index = layer_index
        self._delegate: AdaptiveKVLayerRuntime | HybridStateArraysLayerCache | None = None

    def _resolve_delegate(self) -> AdaptiveKVLayerRuntime | HybridStateArraysLayerCache:
        started_ns = time.perf_counter_ns()
        if self._delegate is not None:
            self._manager.record_perf_ns(
                "family.hybrid_delegate_resolve_ns",
                time.perf_counter_ns() - started_ns,
            )
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
            self._delegate = HybridStateArraysLayerCache()
        else:
            backend = TurboQuantResidentBackend(
                safe_bits=self._manager.config.tq_safe_bits,
                aggr_bits=self._manager.config.tq_aggr_bits,
                safe_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
                aggr_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
                perf_trace=self._manager.perf_trace,
            )
            self._delegate = HybridStateKVAdaptiveLayerCache(
                self._manager,
                layer_index=self._layer_index,
                backend=backend,
            )
        self._manager.record_perf_ns(
            "family.hybrid_delegate_resolve_ns",
            time.perf_counter_ns() - started_ns,
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

    def degrade_block(self, block_id: int) -> None:
        self._resolve_delegate().degrade_block(block_id)

    def restore_block(self, block_id: int) -> None:
        self._resolve_delegate().restore_block(block_id)

    def evict_block(self, block_id: int) -> None:
        self._resolve_delegate().evict_block(block_id)

    def recover_blocks_from_scratch(
        self,
        blocks: tuple[Any, ...],
        replay_layer: Any,
        replay_backend: AdaptiveKVReplayBackend,
        *,
        recovery_profile: ResidentProfile,
    ) -> None:
        self._resolve_delegate().recover_blocks_from_scratch(
            blocks,
            replay_layer,
            replay_backend,
            recovery_profile=recovery_profile,
        )

    def resident_state_for_execution(self, *, query_tokens: int = 1) -> Any:
        return self._resolve_delegate().resident_state_for_execution(query_tokens=query_tokens)

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

    def required_history_start(self, history_tokens: int) -> int:
        return self._resolve_delegate().required_history_start(history_tokens)

    def record_usage_from_attention(
        self,
        resident_state: Any,
        usage_by_token: mx.array,
    ) -> None:
        self._resolve_delegate().record_usage_from_attention(resident_state, usage_by_token)

    def block_live_bytes(self, block_id: int) -> int:
        return self._resolve_delegate().block_live_bytes(block_id)

    def record_perf_ns(self, name: str, elapsed_ns: int) -> None:
        recorder = getattr(self._resolve_delegate(), "record_perf_ns", None)
        if recorder is not None:
            recorder(name, elapsed_ns)

    def perf_sync_enabled(self) -> bool:
        perf_sync = getattr(self._resolve_delegate(), "perf_sync_enabled", None)
        if perf_sync is not None:
            return bool(perf_sync())
        return False

    def begin_cold_mutation_batch(self) -> None:
        begin = getattr(self._resolve_delegate(), "begin_cold_mutation_batch", None)
        if begin is not None:
            begin()

    def end_cold_mutation_batch(self) -> None:
        end = getattr(self._resolve_delegate(), "end_cold_mutation_batch", None)
        if end is not None:
            end()

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
    def make_layer_runtime(
        self,
        manager: AdaptiveKVManager,
        layer_index: int,
    ) -> HybridStateAdaptiveLayerCache:
        return HybridStateAdaptiveLayerCache(manager, layer_index=layer_index)


class HybridStateReplayBackend(FullAttentionKVReplayBackend):
    """Hybrid-state replay keeps exact model cache layout during scratch replay."""

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
    ):
        if scratch_cache is None:
            scratch_cache = model.make_cache()
            replayed_tokens = 0
            materialized = True
        return super().ensure_scratch_replay_prefix(
            model=model,
            num_layers=num_layers,
            scratch_cache=scratch_cache,
            replayed_tokens=replayed_tokens,
            materialized=materialized,
            source_tokens=source_tokens,
            total_tokens=total_tokens,
            prefill_step_size=prefill_step_size,
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
    "HybridStateReplayBackend",
    "HybridStateRuntimeSubstrate",
    "make_hybrid_state_family_bindings",
]
