"""Family B runtime substrate: TurboQuant-first windowed/sliding KV decoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx

from mlxs.adaptive_kv.families.full_kv import (
    FullAttentionKVAdaptiveLayerCache,
    FullAttentionKVRuntimeSubstrate,
)
from mlxs.adaptive_kv.resident import ResidentExecutionMode, TurboQuantResidentBackend
from mlxs.adaptive_kv.runtime import (
    AdaptiveKVReplayBackend,
    RuntimeFamily,
    RuntimeFamilyBindings,
    RuntimeFamilyDescriptor,
    ScratchReplayState,
)
from mlxs.adaptive_kv.storage import ResidentStateView
from mlxs.cache.attention_mask import create_causal_mask
from mlxs.cache.kv import KVCache

if TYPE_CHECKING:
    from mlxs.adaptive_kv.manager import AdaptiveKVManager


WINDOWED_KV_FAMILY = RuntimeFamilyDescriptor(
    family=RuntimeFamily.WINDOWED_KV,
    display_name="TurboQuant Windowed / Local KV Decoder",
    summary=(
        "Token-addressable TurboQuant resident history with explicit resident logical span, "
        "effective visible span, and window-local replay semantics."
    ),
    token_addressable=True,
    sliding_window=True,
)


class WindowedScratchKVCache(KVCache):
    __slots__ = ("_window_size",)

    def __init__(self, *, window_size: int) -> None:
        super().__init__()
        self._window_size = window_size

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        del return_array
        effective_window = window_size if window_size is not None else self._window_size
        return create_causal_mask(n, offset=self.offset, window_size=effective_window)


class WindowedKVAdaptiveLayerCache(FullAttentionKVAdaptiveLayerCache):
    """Family B layer runtime with window-aware execution materialization."""

    def _configured_window_size(self) -> int | None:
        model = self._manager._model
        if model is None:
            return None
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or not (0 <= self._layer_index < len(layers)):
            return None
        layer = layers[self._layer_index]
        if not getattr(layer, "use_sliding", False):
            return None
        return getattr(getattr(model, "args", None), "sliding_window", None)

    def required_history_start(self, history_tokens: int) -> int:
        window_size = self._configured_window_size()
        if window_size is None:
            return 0
        return max(0, history_tokens - max(0, window_size - 1))

    def _resident_view_query_key_for(self, *, query_tokens: int) -> Any:
        del query_tokens
        return self._configured_window_size()

    def _resident_visible_start_for_query(self, *, query_tokens: int) -> int:
        window_size = self._configured_window_size()
        if window_size is None:
            return 0
        return max(0, self._logical_offset - query_tokens - max(0, window_size - 1))

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        effective_window = (
            window_size if window_size is not None else self._configured_window_size()
        )
        if effective_window is None:
            return super().make_mask(n, return_array=return_array, window_size=None)
        del return_array
        visible_offset = min(self._logical_offset, max(0, effective_window - 1))
        return create_causal_mask(n, offset=visible_offset, window_size=effective_window)

    def _query_resident_state(
        self,
        ordered: tuple[Any, ...],
        *,
        query_tokens: int,
    ) -> ResidentStateView:
        window_size = self._configured_window_size()
        if window_size is None:
            return super()._query_resident_state(ordered, query_tokens=query_tokens)
        return self._backend.query_window_view(
            ordered,
            logical_offset=self._logical_offset,
            query_tokens=query_tokens,
            window_size=window_size,
            topology_epoch=self._topology_epoch,
            tail_epoch=self._tail_epoch,
            execution_view_topology_rebuilds_total=self._execution_view_topology_rebuilds_total,
        )


class WindowedKVRuntimeSubstrate(FullAttentionKVRuntimeSubstrate):
    """Runtime substrate for TurboQuant-first windowed-KV decoder families."""

    def make_layer_runtime(
        self,
        manager: AdaptiveKVManager,
        layer_index: int,
    ) -> WindowedKVAdaptiveLayerCache:
        backend = TurboQuantResidentBackend(
            safe_bits=manager.config.tq_safe_bits,
            aggr_bits=manager.config.tq_aggr_bits,
            safe_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
            aggr_execution_mode=ResidentExecutionMode.DEQUANTIZE_ON_READ,
        )
        return WindowedKVAdaptiveLayerCache(manager, layer_index=layer_index, backend=backend)


class WindowedKVReplayBackend(AdaptiveKVReplayBackend):
    """Replay backend for models mixing full and sliding KV layers."""

    @staticmethod
    def _make_scratch_cache(model: Any, num_layers: int) -> list[Any]:
        layers = getattr(getattr(model, "model", None), "layers", None)
        window_size = getattr(getattr(model, "args", None), "sliding_window", None)
        if layers is None or len(layers) != num_layers:
            return [KVCache() for _ in range(num_layers)]
        out: list[Any] = []
        for layer in layers:
            if getattr(layer, "use_sliding", False) and window_size is not None:
                out.append(WindowedScratchKVCache(window_size=window_size))
            else:
                out.append(KVCache())
        return out

    @staticmethod
    def _cache_eval_tensors(cache: Any) -> list[mx.array]:
        state = getattr(cache, "state", None)
        if state is None:
            return []
        if isinstance(state, tuple):
            return [tensor for tensor in state if hasattr(tensor, "nbytes")]
        if hasattr(state, "nbytes"):
            return [state]
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
            scratch_cache = self._make_scratch_cache(model, num_layers)
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


def make_windowed_kv_family_bindings() -> RuntimeFamilyBindings:
    return RuntimeFamilyBindings(
        descriptor=WINDOWED_KV_FAMILY,
        runtime_substrate=WindowedKVRuntimeSubstrate(),
        replay_backend=WindowedKVReplayBackend(),
        layer_runtime_type=WindowedKVAdaptiveLayerCache,
    )


__all__ = [
    "WINDOWED_KV_FAMILY",
    "WindowedKVAdaptiveLayerCache",
    "WindowedKVReplayBackend",
    "WindowedKVRuntimeSubstrate",
    "make_windowed_kv_family_bindings",
]
