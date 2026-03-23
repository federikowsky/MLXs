"""Generic Adaptive KV runtime contracts, capabilities, and platform composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import mlx.core as mx


class SupportLevel(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    level: SupportLevel
    reason: str | None = None

    @classmethod
    def full(cls) -> CapabilityStatus:
        return cls(SupportLevel.FULL)

    @classmethod
    def partial(cls, reason: str) -> CapabilityStatus:
        return cls(SupportLevel.PARTIAL, reason=reason)

    @classmethod
    def unsupported(cls, reason: str) -> CapabilityStatus:
        return cls(SupportLevel.UNSUPPORTED, reason=reason)

    @property
    def supported(self) -> bool:
        return self.level is SupportLevel.FULL


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    adapter_name: str
    overall: CapabilityStatus
    baseline_cache: CapabilityStatus
    resident_attention: CapabilityStatus
    compressed_tier: CapabilityStatus
    replay_recovery: CapabilityStatus
    num_layers: int = 0

    @property
    def supported(self) -> bool:
        return self.overall.supported

    @property
    def reason(self) -> str | None:
        return self.overall.reason


@dataclass(frozen=True, slots=True)
class ScratchReplayState:
    cache: list[Any]
    replayed_tokens: int
    materialized: bool


@runtime_checkable
class AdaptiveKVLayerRuntime(Protocol):
    @property
    def offset(self) -> int: ...

    @property
    def keys(self) -> mx.array | None: ...

    @property
    def values(self) -> mx.array | None: ...

    @property
    def state(self) -> tuple[Any, ...] | None: ...

    @property
    def live_state_size_bytes(self) -> int: ...

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[Any, Any]: ...

    def remove_token_range(self, start: int, end: int) -> None: ...

    def demote_block(self, block_id: int) -> None: ...

    def promote_block(self, block_id: int) -> None: ...

    def evict_block(self, block_id: int) -> None: ...

    def recover_block(self, block_id: int, keys: mx.array, values: mx.array) -> None: ...

    def recover_blocks(
        self,
        blocks: tuple[Any, ...],
        keys: mx.array,
        values: mx.array,
    ) -> None: ...

    def resident_state_for_attention(self) -> Any: ...

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None: ...

    def reset(self) -> None: ...

    def trim(self, n: int) -> int: ...

    def should_sample_usage(self) -> bool: ...

    def record_usage_from_attention(
        self,
        resident_state: Any,
        usage_by_token: mx.array,
    ) -> None: ...

    def block_live_bytes(self, block_id: int) -> int: ...


@runtime_checkable
class AdaptiveKVCapabilityProvider(Protocol):
    def matches_model(self, model: Any) -> bool: ...

    def assess_generation_support(
        self,
        model: Any,
        *,
        cache: list[Any] | None,
        compile_decode: bool,
        quantized_kv_start: int,
        input_embeddings_present: bool = False,
    ) -> AdapterCapabilities: ...


@runtime_checkable
class AdaptiveKVRuntimeSubstrate(Protocol):
    def make_layer_runtime(self, manager: Any, layer_index: int) -> AdaptiveKVLayerRuntime: ...


@runtime_checkable
class AdaptiveKVReplayBackend(Protocol):
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
    ) -> ScratchReplayState: ...

    def copy_replay_token_range(
        self,
        replay_layer: Any,
        start_token: int,
        end_token: int,
    ) -> tuple[mx.array, mx.array]: ...


@runtime_checkable
class AdaptiveKVRuntimeAdapter(Protocol):
    name: str
    capability_provider: AdaptiveKVCapabilityProvider
    runtime_substrate: AdaptiveKVRuntimeSubstrate
    replay_backend: AdaptiveKVReplayBackend
    layer_runtime_type: type[Any] | None

    def matches_model(self, model: Any) -> bool: ...

    def assess_generation_support(
        self,
        model: Any,
        *,
        cache: list[Any] | None,
        compile_decode: bool,
        quantized_kv_start: int,
        input_embeddings_present: bool = False,
    ) -> AdapterCapabilities: ...


class ComposedAdaptiveKVRuntimeAdapter:
    """Concrete adapter assembled from independent capability/runtime components."""

    __slots__ = (
        "capability_provider",
        "layer_runtime_type",
        "name",
        "replay_backend",
        "runtime_substrate",
    )

    def __init__(
        self,
        *,
        name: str,
        capability_provider: AdaptiveKVCapabilityProvider,
        runtime_substrate: AdaptiveKVRuntimeSubstrate,
        replay_backend: AdaptiveKVReplayBackend,
        layer_runtime_type: type[Any] | None = None,
    ) -> None:
        self.name = name
        self.capability_provider = capability_provider
        self.runtime_substrate = runtime_substrate
        self.replay_backend = replay_backend
        self.layer_runtime_type = layer_runtime_type

    def matches_model(self, model: Any) -> bool:
        return self.capability_provider.matches_model(model)

    def assess_generation_support(
        self,
        model: Any,
        *,
        cache: list[Any] | None,
        compile_decode: bool,
        quantized_kv_start: int,
        input_embeddings_present: bool = False,
    ) -> AdapterCapabilities:
        return self.capability_provider.assess_generation_support(
            model,
            cache=cache,
            compile_decode=compile_decode,
            quantized_kv_start=quantized_kv_start,
            input_embeddings_present=input_embeddings_present,
        )

    def make_layer_runtime(self, manager: Any, layer_index: int) -> AdaptiveKVLayerRuntime:
        return self.runtime_substrate.make_layer_runtime(manager, layer_index)

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
        return self.replay_backend.ensure_scratch_replay_prefix(
            model=model,
            num_layers=num_layers,
            scratch_cache=scratch_cache,
            replayed_tokens=replayed_tokens,
            materialized=materialized,
            source_tokens=source_tokens,
            total_tokens=total_tokens,
            prefill_step_size=prefill_step_size,
        )

    def copy_replay_token_range(
        self,
        replay_layer: Any,
        start_token: int,
        end_token: int,
    ) -> tuple[mx.array, mx.array]:
        return self.replay_backend.copy_replay_token_range(
            replay_layer,
            start_token,
            end_token,
        )


@dataclass(frozen=True, slots=True)
class AdapterSelection:
    adapter: AdaptiveKVRuntimeAdapter | None
    capabilities: AdapterCapabilities

    @property
    def platform(self) -> AdaptiveKVRuntimeAdapter | None:
        return self.adapter

    @property
    def num_layers(self) -> int:
        return self.capabilities.num_layers

    @property
    def runtime_substrate(self) -> AdaptiveKVRuntimeSubstrate | None:
        if self.adapter is None:
            return None
        return self.adapter.runtime_substrate

    @property
    def replay_backend(self) -> AdaptiveKVReplayBackend | None:
        if self.adapter is None:
            return None
        return self.adapter.replay_backend
