"""Qwen 3.5 adapter layered on runtime-family classification."""

from __future__ import annotations

from typing import Any

from mlxs.adaptive_kv.block_types import ResidentProfile
from mlxs.adaptive_kv.families import (
    make_full_kv_family_bindings,
    make_hybrid_state_family_bindings,
)
from mlxs.adaptive_kv.families.full_kv import FULL_KV_FAMILY
from mlxs.adaptive_kv.resident import ResidentExecutionMode
from mlxs.adaptive_kv.runtime import (
    AdapterCapabilities,
    AdaptiveKVCapabilityProvider,
    CapabilityStatus,
    ComposedAdaptiveKVRuntimeAdapter,
    FamilyProfileCapability,
    RuntimeFamily,
)
from mlxs.cache.arrays import ArraysCache
from mlxs.cache.kv import KVCache


def _capabilities(
    *,
    adapter_name: str,
    runtime_family: RuntimeFamily,
    overall: CapabilityStatus,
    num_layers: int = 0,
) -> AdapterCapabilities:
    execution_modes = (ResidentExecutionMode.DEQUANTIZE_ON_READ,)
    profile_capabilities = (
        FamilyProfileCapability(
            profile=ResidentProfile.TQ_SAFE,
            status=overall,
            execution_modes=execution_modes,
        ),
        FamilyProfileCapability(
            profile=ResidentProfile.TQ_AGGR,
            status=overall,
            execution_modes=execution_modes,
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


class Qwen35CapabilityProvider(AdaptiveKVCapabilityProvider):
    """Support assessment for qwen3_5 under the retained runtime-family baseline."""

    name = "qwen3_5"

    def matches_model(self, model: Any) -> bool:
        return getattr(model, "model_type", None) == "qwen3_5"

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
                    "adaptive_kv_turboquant supports model_type='qwen3_5' only through the "
                    "dedicated qwen3_5 adapter"
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
                runtime_family=RuntimeFamily.HYBRID_STATE,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant supports text-only qwen3_5 requests only; "
                    "multimodal/input_embeddings requests are unsupported"
                ),
            )
        if not callable(getattr(model, "make_cache", None)):
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant requires model.make_cache() for the "
                    "supported qwen3_5 baseline"
                ),
            )

        args = getattr(model, "args", None)
        full_attention_interval = getattr(args, "full_attention_interval", None)
        runtime_family = (
            RuntimeFamily.FULL_KV
            if full_attention_interval in (None, 1)
            else RuntimeFamily.HYBRID_STATE
        )

        head_dim = getattr(args, "head_dim", None)
        if head_dim is None and args is not None:
            hidden_size = getattr(args, "hidden_size", None)
            num_attention_heads = getattr(args, "num_attention_heads", None)
            if hidden_size is not None and num_attention_heads:
                head_dim = hidden_size // num_attention_heads
        if head_dim is not None and head_dim % 32 != 0:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant requires qwen3_5 head_dim divisible by 32 "
                    "for TurboQuant resident profiles"
                ),
            )

        model_impl = getattr(model, "model", None)
        layers = getattr(model_impl, "layers", None)
        if layers is not None and any(getattr(layer, "is_linear", False) for layer in layers):
            runtime_family = RuntimeFamily.HYBRID_STATE

        baseline = model.make_cache()
        if not isinstance(baseline, list) or not baseline:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=runtime_family,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_turboquant requires a non-empty per-layer cache list"
                ),
            )
        if runtime_family is RuntimeFamily.FULL_KV:
            if any(type(layer) is not KVCache for layer in baseline):
                return _capabilities(
                    adapter_name=self.name,
                    runtime_family=RuntimeFamily.FULL_KV,
                    overall=CapabilityStatus.partial(
                        "adaptive_kv_turboquant supports full_kv qwen3_5 only for a homogeneous "
                        "list[KVCache] baseline"
                    ),
                )
        else:
            if layers is None or len(layers) != len(baseline):
                return _capabilities(
                    adapter_name=self.name,
                    runtime_family=RuntimeFamily.HYBRID_STATE,
                    overall=CapabilityStatus.partial(
                        "adaptive_kv_turboquant requires layer/cache alignment for hybrid-state "
                        "qwen3_5 support"
                    ),
                )
            for layer, cache_entry in zip(layers, baseline, strict=True):
                if getattr(layer, "is_linear", False):
                    if type(cache_entry) is not ArraysCache:
                        return _capabilities(
                            adapter_name=self.name,
                            runtime_family=RuntimeFamily.HYBRID_STATE,
                            overall=CapabilityStatus.partial(
                                "adaptive_kv_turboquant hybrid-state qwen3_5 support requires "
                                "ArraysCache on linear-attention layers"
                            ),
                        )
                elif type(cache_entry) is not KVCache:
                    return _capabilities(
                        adapter_name=self.name,
                        runtime_family=RuntimeFamily.HYBRID_STATE,
                        overall=CapabilityStatus.partial(
                            "adaptive_kv_turboquant hybrid-state qwen3_5 support requires KVCache "
                            "on full-attention layers"
                        ),
                    )

        return _capabilities(
            adapter_name=self.name,
            runtime_family=runtime_family,
            overall=CapabilityStatus.full(),
            num_layers=len(baseline),
        )


class Qwen35AdaptiveKVAdapter(ComposedAdaptiveKVRuntimeAdapter):
    """Concrete adapter for the retained qwen3_5 full-attention-only subset."""

    def __init__(self) -> None:
        super().__init__(
            name="qwen3_5",
            capability_provider=Qwen35CapabilityProvider(),
            default_family=FULL_KV_FAMILY.family,
            family_bindings=(
                make_full_kv_family_bindings(),
                make_hybrid_state_family_bindings(),
            ),
        )


__all__ = ["Qwen35AdaptiveKVAdapter", "Qwen35CapabilityProvider"]
