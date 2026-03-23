"""Compatibility gates and adapter selection for Adaptive KV V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlxs.adaptive_kv.adapters import resolve_generation_adapter
from mlxs.adaptive_kv.runtime import (
    AdapterCapabilities,
    AdapterSelection,
    CapabilityStatus,
    SupportLevel,
)


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    supported: bool
    reason: str | None = None
    num_layers: int = 0
    adapter_name: str | None = None
    support_level: SupportLevel = SupportLevel.UNSUPPORTED
    capabilities: AdapterCapabilities | None = None


def select_generation_adapter(
    model: Any,
    *,
    cache: list[Any] | None,
    compile_decode: bool,
    quantized_kv_start: int,
    input_embeddings_present: bool = False,
) -> AdapterSelection:
    adapter = resolve_generation_adapter(model)
    if adapter is None:
        capabilities = AdapterCapabilities(
            adapter_name=getattr(model, "model_type", None) or "unknown",
            overall=CapabilityStatus.unsupported(
                "adaptive_kv_v1 supports model_type='llama' only"
            ),
            baseline_cache=CapabilityStatus.unsupported(
                "adaptive_kv_v1 supports model_type='llama' only"
            ),
            resident_attention=CapabilityStatus.unsupported(
                "adaptive_kv_v1 supports model_type='llama' only"
            ),
            compressed_tier=CapabilityStatus.unsupported(
                "adaptive_kv_v1 supports model_type='llama' only"
            ),
            replay_recovery=CapabilityStatus.unsupported(
                "adaptive_kv_v1 supports model_type='llama' only"
            ),
            num_layers=0,
        )
        return AdapterSelection(adapter=None, capabilities=capabilities)
    capabilities = adapter.assess_generation_support(
        model,
        cache=cache,
        compile_decode=compile_decode,
        quantized_kv_start=quantized_kv_start,
        input_embeddings_present=input_embeddings_present,
    )
    return AdapterSelection(
        adapter=adapter if capabilities.supported else None,
        capabilities=capabilities,
    )


def assess_generation_compatibility(
    model: Any,
    *,
    cache: list[Any] | None,
    compile_decode: bool,
    quantized_kv_start: int,
    input_embeddings_present: bool = False,
) -> CompatibilityResult:
    """Return whether adaptive KV V1 supports this generation request."""
    selection = select_generation_adapter(
        model,
        cache=cache,
        compile_decode=compile_decode,
        quantized_kv_start=quantized_kv_start,
        input_embeddings_present=input_embeddings_present,
    )
    capabilities = selection.capabilities
    return CompatibilityResult(
        supported=capabilities.supported,
        reason=capabilities.reason,
        num_layers=capabilities.num_layers,
        adapter_name=capabilities.adapter_name,
        support_level=capabilities.overall.level,
        capabilities=capabilities,
    )
