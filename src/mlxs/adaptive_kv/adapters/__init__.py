"""Adaptive KV runtime adapters."""

from __future__ import annotations

from typing import Any

from mlxs.adaptive_kv.adapters.llama import LlamaAdaptiveKVAdapter, LlamaAdaptiveLayerCache
from mlxs.adaptive_kv.runtime import AdaptiveKVRuntimeAdapter

_GENERATION_ADAPTERS: tuple[AdaptiveKVRuntimeAdapter, ...] = (LlamaAdaptiveKVAdapter(),)


def generation_adapters() -> tuple[AdaptiveKVRuntimeAdapter, ...]:
    return _GENERATION_ADAPTERS


def resolve_generation_adapter(model: Any) -> AdaptiveKVRuntimeAdapter | None:
    for adapter in _GENERATION_ADAPTERS:
        if adapter.matches_model(model):
            return adapter
    return None


__all__ = [
    "LlamaAdaptiveKVAdapter",
    "LlamaAdaptiveLayerCache",
    "generation_adapters",
    "resolve_generation_adapter",
]
