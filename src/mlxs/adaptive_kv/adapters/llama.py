"""Retained llama adapter layered on the generic full-KV runtime family."""

from __future__ import annotations

from typing import Any

from mlxs.adaptive_kv.block_types import ResidentProfile
from mlxs.adaptive_kv.families import (
    FullAttentionKVAdaptiveLayerCache,
    FullAttentionKVReplayBackend,
    FullAttentionKVRuntimeSubstrate,
    make_full_kv_family_bindings,
)
from mlxs.adaptive_kv.families.full_kv import FULL_KV_FAMILY
from mlxs.adaptive_kv.families.windowed_kv import WINDOWED_KV_FAMILY
from mlxs.adaptive_kv.resident import ResidentExecutionMode
from mlxs.adaptive_kv.runtime import (
    AdapterCapabilities,
    AdaptiveKVCapabilityProvider,
    CapabilityStatus,
    ComposedAdaptiveKVRuntimeAdapter,
    FamilyProfileCapability,
    RuntimeFamily,
)
from mlxs.cache.kv import KVCache


def _capabilities(
    *,
    adapter_name: str,
    runtime_family: RuntimeFamily,
    overall: CapabilityStatus,
    num_layers: int = 0,
) -> AdapterCapabilities:
    profile_capabilities = (
        FamilyProfileCapability(
            profile=ResidentProfile.TQ_SAFE,
            status=overall,
            execution_modes=(ResidentExecutionMode.DEQUANTIZE_ON_READ,),
        ),
        FamilyProfileCapability(
            profile=ResidentProfile.TQ_AGGR,
            status=overall,
            execution_modes=(ResidentExecutionMode.DEQUANTIZE_ON_READ,),
        ),
        FamilyProfileCapability(
            profile=ResidentProfile.EVICTED,
            status=CapabilityStatus.full(),
        ),
    )
    return AdapterCapabilities(
        adapter_name=adapter_name,
        runtime_family=runtime_family,
        overall=overall,
        baseline_cache=overall,
        resident_attention=overall,
        replay_recovery=overall,
        profile_capabilities=profile_capabilities,
        num_layers=num_layers,
    )


class LlamaCapabilityProvider(AdaptiveKVCapabilityProvider):
    """Support assessment for the retained Llama runtime family envelope."""

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
                runtime_family=RuntimeFamily.UNKNOWN,
                overall=CapabilityStatus.unsupported(
                    "adaptive_kv_turboquant supports model_type='llama' only through "
                    "the dedicated llama adapter"
                ),
            )
        if compile_decode:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant does not support compile_decode=True"
                ),
            )
        if quantized_kv_start > 0:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant does not support legacy quantized_kv_start flow"
                ),
            )
        if cache is not None:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant does not support external cache reuse"
                ),
            )
        if input_embeddings_present:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant does not support multimodal/input_embeddings requests"
                ),
            )
        if not callable(getattr(model, "make_cache", None)):
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant requires model.make_cache() for the "
                    "supported llama baseline"
                ),
            )

        args = getattr(model, "args", None)
        layer_types = getattr(args, "layer_types", None)
        if layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.WINDOWED_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant supports the standard full-attention "
                    "llama baseline only; "
                    f"{WINDOWED_KV_FAMILY.display_name.lower()} layer_types are unsupported"
                ),
            )

        model_impl = getattr(model, "model", None)
        layers = getattr(model_impl, "layers", None)
        if layers is not None and any(getattr(layer, "use_sliding", False) for layer in layers):
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.WINDOWED_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant supports the standard full-attention "
                    "llama baseline only; "
                    "sliding-window llama layers fall into the windowed-KV runtime family "
                    "and remain unsupported"
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
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant requires llama head_dim divisible by 32 "
                    "for TurboQuant resident profiles"
                ),
            )

        baseline = model.make_cache()
        if not isinstance(baseline, list) or not baseline:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant requires a non-empty per-layer cache list"
                ),
            )
        if any(type(layer) is not KVCache for layer in baseline):
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.WINDOWED_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant requires a homogeneous list[KVCache] baseline"
                ),
            )
        return _capabilities(
            adapter_name=self.name,
            runtime_family=RuntimeFamily.FULL_KV,
            overall=CapabilityStatus.full(),
            num_layers=len(baseline),
        )


class LlamaAdaptiveKVAdapter(ComposedAdaptiveKVRuntimeAdapter):
    """Concrete platform adapter for the retained full-attention llama baseline."""

    def __init__(self) -> None:
        super().__init__(
            name="llama",
            capability_provider=LlamaCapabilityProvider(),
            default_family=FULL_KV_FAMILY.family,
            family_bindings=(make_full_kv_family_bindings(),),
        )


LlamaAdaptiveLayerCache = FullAttentionKVAdaptiveLayerCache
LlamaRuntimeSubstrate = FullAttentionKVRuntimeSubstrate
LlamaReplayBackend = FullAttentionKVReplayBackend


__all__ = [
    "LlamaAdaptiveKVAdapter",
    "LlamaAdaptiveLayerCache",
    "LlamaCapabilityProvider",
    "LlamaReplayBackend",
    "LlamaRuntimeSubstrate",
]
