"""Ministral3 adapter with explicit runtime-family classification."""

from __future__ import annotations

from typing import Any

from mlxs.adaptive_kv.families import (
    make_full_kv_family_bindings,
    make_windowed_kv_family_bindings,
)
from mlxs.adaptive_kv.families.full_kv import FULL_KV_FAMILY
from mlxs.adaptive_kv.runtime import (
    AdapterCapabilities,
    AdaptiveKVCapabilityProvider,
    CapabilityStatus,
    ComposedAdaptiveKVRuntimeAdapter,
    RuntimeFamily,
)
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache


def _capabilities(
    *,
    adapter_name: str,
    runtime_family: RuntimeFamily,
    overall: CapabilityStatus,
    num_layers: int = 0,
) -> AdapterCapabilities:
    return AdapterCapabilities(
        adapter_name=adapter_name,
        runtime_family=runtime_family,
        overall=overall,
        baseline_cache=overall,
        resident_attention=overall,
        compressed_tier=overall,
        replay_recovery=overall,
        num_layers=num_layers,
    )


class Ministral3CapabilityProvider(AdaptiveKVCapabilityProvider):
    """Support assessment for Ministral3 under the retained runtime-family baseline."""

    name = "ministral3"

    def matches_model(self, model: Any) -> bool:
        return getattr(model, "model_type", None) == "ministral3"

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
                    "adaptive_kv_v1 supports model_type='ministral3' only through the "
                    "dedicated ministral3 adapter"
                ),
            )
        if compile_decode:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 does not support compile_decode=True"
                ),
            )
        if quantized_kv_start > 0:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 does not support legacy quantized_kv_start flow"
                ),
            )
        if cache is not None:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 does not support external cache reuse"
                ),
            )
        if input_embeddings_present:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 does not support multimodal/input_embeddings requests"
                ),
            )
        if not callable(getattr(model, "make_cache", None)):
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.FULL_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 requires model.make_cache() for the supported "
                    "ministral3 baseline"
                ),
            )

        args = getattr(model, "args", None)
        layer_types = getattr(args, "layer_types", None)
        runtime_family = RuntimeFamily.FULL_KV
        if layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            runtime_family = RuntimeFamily.WINDOWED_KV

        model_impl = getattr(model, "model", None)
        layers = getattr(model_impl, "layers", None)
        if layers is not None and any(getattr(layer, "use_sliding", False) for layer in layers):
            runtime_family = RuntimeFamily.WINDOWED_KV
        if runtime_family is RuntimeFamily.WINDOWED_KV and getattr(
            args, "sliding_window", None
        ) is None:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=RuntimeFamily.WINDOWED_KV,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 windowed_kv ministral3 support requires sliding_window "
                    "to be configured for sliding layers"
                ),
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
                    "adaptive_kv_v1 compressed tier requires ministral3 head_dim divisible by 32"
                ),
            )

        baseline = model.make_cache()
        if not isinstance(baseline, list) or not baseline:
            return _capabilities(
                adapter_name=self.name,
                runtime_family=runtime_family,
                overall=CapabilityStatus.partial(
                    "adaptive_kv_v1 requires a non-empty per-layer cache list"
                ),
            )
        if runtime_family is RuntimeFamily.FULL_KV:
            if any(type(layer) is not KVCache for layer in baseline):
                return _capabilities(
                    adapter_name=self.name,
                    runtime_family=RuntimeFamily.FULL_KV,
                    overall=CapabilityStatus.partial(
                        "adaptive_kv_v1 supports full_kv ministral3 only for a homogeneous "
                        "list[KVCache] baseline"
                    ),
                )
        else:
            if layers is None or len(layers) != len(baseline):
                return _capabilities(
                    adapter_name=self.name,
                    runtime_family=RuntimeFamily.WINDOWED_KV,
                    overall=CapabilityStatus.partial(
                        "adaptive_kv_v1 requires layer/cache alignment for windowed_kv "
                        "ministral3 support"
                    ),
                )
            for layer, cache_entry in zip(layers, baseline, strict=True):
                if getattr(layer, "use_sliding", False):
                    if type(cache_entry) is not RotatingKVCache:
                        return _capabilities(
                            adapter_name=self.name,
                            runtime_family=RuntimeFamily.WINDOWED_KV,
                            overall=CapabilityStatus.partial(
                                "adaptive_kv_v1 windowed_kv ministral3 support requires "
                                "RotatingKVCache on sliding layers"
                            ),
                        )
                elif type(cache_entry) is not KVCache:
                    return _capabilities(
                        adapter_name=self.name,
                        runtime_family=RuntimeFamily.WINDOWED_KV,
                        overall=CapabilityStatus.partial(
                            "adaptive_kv_v1 windowed_kv ministral3 support requires KVCache "
                            "on full-attention layers"
                        ),
                    )
        return _capabilities(
            adapter_name=self.name,
            runtime_family=runtime_family,
            overall=CapabilityStatus.full(),
            num_layers=len(baseline),
        )


class Ministral3AdaptiveKVAdapter(ComposedAdaptiveKVRuntimeAdapter):
    """Concrete adapter for Ministral3 runtime-family classification."""

    def __init__(self) -> None:
        super().__init__(
            name="ministral3",
            capability_provider=Ministral3CapabilityProvider(),
            default_family=FULL_KV_FAMILY.family,
            family_bindings=(
                make_full_kv_family_bindings(),
                make_windowed_kv_family_bindings(),
            ),
        )


__all__ = ["Ministral3AdaptiveKVAdapter", "Ministral3CapabilityProvider"]
