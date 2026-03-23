"""Compatibility gates and adapter selection for Adaptive KV V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlxs.adaptive_kv.adapters import resolve_generation_adapter
from mlxs.adaptive_kv.runtime import (
    AdapterCapabilities,
    AdapterSelection,
    CapabilityStatus,
    RuntimeFamily,
    SupportLevel,
)


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    supported: bool
    reason: str | None = None
    num_layers: int = 0
    adapter_name: str | None = None
    runtime_family: RuntimeFamily = RuntimeFamily.UNKNOWN
    support_level: SupportLevel = SupportLevel.UNSUPPORTED
    capabilities: AdapterCapabilities | None = None


def _unsupported_capabilities(
    *,
    adapter_name: str,
    reason: str,
    runtime_family: RuntimeFamily = RuntimeFamily.UNKNOWN,
) -> AdapterCapabilities:
    unsupported = CapabilityStatus.unsupported(reason)
    return AdapterCapabilities(
        adapter_name=adapter_name,
        runtime_family=runtime_family,
        overall=unsupported,
        baseline_cache=unsupported,
        resident_attention=unsupported,
        compressed_tier=unsupported,
        replay_recovery=unsupported,
        num_layers=0,
    )


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
        model_type = getattr(model, "model_type", None) or "unknown"
        capabilities = _unsupported_capabilities(
            adapter_name=model_type,
            reason=(
                "adaptive_kv_v1 has no registered exact adapter for "
                f"model_type={model_type!r}"
            ),
        )
        return AdapterSelection(adapter=None, capabilities=capabilities, family_bindings=None)
    capabilities = adapter.assess_generation_support(
        model,
        cache=cache,
        compile_decode=compile_decode,
        quantized_kv_start=quantized_kv_start,
        input_embeddings_present=input_embeddings_present,
    )
    family_bindings = None
    selected_adapter = adapter if capabilities.supported else None
    if capabilities.supported:
        family_bindings = adapter.resolve_family_bindings(capabilities.runtime_family)
        if family_bindings is None:
            capabilities = _unsupported_capabilities(
                adapter_name=capabilities.adapter_name,
                runtime_family=capabilities.runtime_family,
                reason=(
                    "adaptive_kv_v1 adapter bindings are incomplete for runtime_family="
                    f"{capabilities.runtime_family.value!r}"
                ),
            )
            selected_adapter = None
    return AdapterSelection(
        adapter=selected_adapter,
        capabilities=capabilities,
        family_bindings=family_bindings,
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
        runtime_family=capabilities.runtime_family,
        support_level=capabilities.overall.level,
        capabilities=capabilities,
    )
