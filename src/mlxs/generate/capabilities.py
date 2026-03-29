"""Internal decode/prefill capability resolution for single-request generate.

The single-request path is mostly uniform across models, but a few important
runtime assumptions vary by cache/model surface. This module resolves those
assumptions once so planning and prefill can be explicit about them.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mlxs.cache import (
    ArraysCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)


class CacheTopology(StrEnum):
    """High-level cache topology relevant to generate/prefill runtime behavior."""

    KV = "kv"
    STATE = "state"
    HYBRID = "hybrid"


class PrefillSyncStrategy(StrEnum):
    """How chunked prefill should force chunk execution before clearing cache."""

    CACHE_STATE = "cache_state"
    MODEL_OUTPUT = "model_output"


@dataclass(frozen=True, slots=True)
class DecodeCapabilities:
    """Resolved internal runtime assumptions for single-request generate."""

    cache_topology: CacheTopology
    supports_delayed_kv_quantization: bool
    prefill_sync_strategy: PrefillSyncStrategy
    supports_prefill_input_embeddings: bool


_KV_CACHE_TYPES = (KVCache, QuantizedKVCache, RotatingKVCache, ChunkedKVCache)
_KNOWN_CACHE_TYPES = (ArraysCache, CacheList, *_KV_CACHE_TYPES)
_DELAYED_KV_UNSUPPORTED_TYPES = (
    ArraysCache,
    CacheList,
    ChunkedKVCache,
    QuantizedKVCache,
    RotatingKVCache,
)


def resolve_decode_capabilities(
    model: Any,
    cache: list[Any],
) -> DecodeCapabilities:
    """Resolve decode/prefill capabilities from the current model + cache reality."""

    return DecodeCapabilities(
        cache_topology=_resolve_cache_topology(cache),
        supports_delayed_kv_quantization=_supports_delayed_kv_quantization(cache),
        prefill_sync_strategy=_resolve_prefill_sync_strategy(cache),
        supports_prefill_input_embeddings=_supports_prefill_input_embeddings(model),
    )


def _resolve_cache_topology(cache: list[Any]) -> CacheTopology:
    if not cache:
        return CacheTopology.KV
    if all(isinstance(layer, ArraysCache) for layer in cache):
        return CacheTopology.STATE
    if all(isinstance(layer, _KV_CACHE_TYPES) for layer in cache):
        return CacheTopology.KV
    if any(isinstance(layer, CacheList) for layer in cache):
        return CacheTopology.HYBRID
    if any(isinstance(layer, _KNOWN_CACHE_TYPES) for layer in cache):
        return CacheTopology.HYBRID
    return CacheTopology.KV


def _supports_delayed_kv_quantization(cache: list[Any]) -> bool:
    if not cache:
        return False
    if all(isinstance(layer, KVCache) for layer in cache):
        return True
    return not any(isinstance(layer, _DELAYED_KV_UNSUPPORTED_TYPES) for layer in cache)


def _resolve_prefill_sync_strategy(cache: list[Any]) -> PrefillSyncStrategy:
    if cache and all(hasattr(layer, "state") for layer in cache):
        return PrefillSyncStrategy.CACHE_STATE
    return PrefillSyncStrategy.MODEL_OUTPUT


def _supports_prefill_input_embeddings(model: Any) -> bool:
    try:
        signature = inspect.signature(model.__call__)
    except (TypeError, ValueError):
        return False
    if "input_embeddings" in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


__all__ = [
    "CacheTopology",
    "DecodeCapabilities",
    "PrefillSyncStrategy",
    "resolve_decode_capabilities",
]
