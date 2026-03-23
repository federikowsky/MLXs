"""Adaptive KV runtime adapter registry."""

from __future__ import annotations

from typing import Any

from mlxs.adaptive_kv.adapters.llama import LlamaAdaptiveKVAdapter, LlamaAdaptiveLayerCache
from mlxs.adaptive_kv.runtime import AdaptiveKVRuntimeAdapter


class AdaptiveKVAdapterRegistry:
    """Small immutable registry for generation-time Adaptive KV adapters."""

    __slots__ = ("_generation_adapters",)

    def __init__(self, generation_adapters: tuple[AdaptiveKVRuntimeAdapter, ...]) -> None:
        if not generation_adapters:
            raise ValueError("Adaptive KV adapter registry requires at least one adapter")
        self._generation_adapters = generation_adapters

    def generation_adapters(self) -> tuple[AdaptiveKVRuntimeAdapter, ...]:
        return self._generation_adapters

    def default_generation_adapter(self) -> AdaptiveKVRuntimeAdapter:
        return self._generation_adapters[0]

    def resolve_generation_adapter(self, model: Any) -> AdaptiveKVRuntimeAdapter | None:
        for adapter in self._generation_adapters:
            if adapter.matches_model(model):
                return adapter
        return None


_GENERATION_REGISTRY = AdaptiveKVAdapterRegistry((LlamaAdaptiveKVAdapter(),))


def generation_adapter_registry() -> AdaptiveKVAdapterRegistry:
    return _GENERATION_REGISTRY


def generation_adapters() -> tuple[AdaptiveKVRuntimeAdapter, ...]:
    return _GENERATION_REGISTRY.generation_adapters()


def default_generation_adapter() -> AdaptiveKVRuntimeAdapter:
    return _GENERATION_REGISTRY.default_generation_adapter()


def resolve_generation_adapter(model: Any) -> AdaptiveKVRuntimeAdapter | None:
    return _GENERATION_REGISTRY.resolve_generation_adapter(model)


__all__ = [
    "AdaptiveKVAdapterRegistry",
    "LlamaAdaptiveKVAdapter",
    "LlamaAdaptiveLayerCache",
    "default_generation_adapter",
    "generation_adapter_registry",
    "generation_adapters",
    "resolve_generation_adapter",
]
