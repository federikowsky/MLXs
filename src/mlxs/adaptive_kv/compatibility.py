"""Compatibility gates for adaptive KV V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlxs.cache.kv import KVCache


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    supported: bool
    reason: str | None = None
    num_layers: int = 0


def assess_generation_compatibility(
    model: Any,
    *,
    cache: list[Any] | None,
    compile_decode: bool,
    quantized_kv_start: int,
    input_embeddings_present: bool = False,
) -> CompatibilityResult:
    """Return whether adaptive KV V1 supports this generation request."""
    if compile_decode:
        return CompatibilityResult(
            supported=False,
            reason="adaptive_kv_v1 does not support compile_decode=True",
        )
    if quantized_kv_start > 0:
        return CompatibilityResult(
            supported=False,
            reason="adaptive_kv_v1 does not support legacy quantized_kv_start flow",
        )
    if cache is not None:
        return CompatibilityResult(
            supported=False,
            reason="adaptive_kv_v1 does not support external cache reuse",
        )
    if input_embeddings_present:
        return CompatibilityResult(
            supported=False,
            reason="adaptive_kv_v1 does not support multimodal/input_embeddings requests",
        )
    if getattr(model, "model_type", None) != "llama":
        return CompatibilityResult(
            supported=False,
            reason="adaptive_kv_v1 supports model_type='llama' only",
        )
    if not callable(getattr(model, "make_cache", None)):
        return CompatibilityResult(
            supported=False,
            reason="adaptive_kv_v1 requires model.make_cache() for the supported llama baseline",
    )
    args = getattr(model, "args", None)
    layer_types = getattr(args, "layer_types", None)
    if layer_types is not None and any(
        layer_type != "full_attention" for layer_type in layer_types
    ):
        return CompatibilityResult(
            supported=False,
            reason=(
                "adaptive_kv_v1 supports the standard full-attention llama baseline only; "
                "sliding or mixed layer_types are unsupported"
            ),
        )
    model_impl = getattr(model, "model", None)
    layers = getattr(model_impl, "layers", None)
    if layers is not None and any(getattr(layer, "use_sliding", False) for layer in layers):
        return CompatibilityResult(
            supported=False,
            reason=(
                "adaptive_kv_v1 supports the standard full-attention llama baseline only; "
                "sliding-window llama layers are unsupported"
            ),
        )
    if args is not None:
        hidden_size = getattr(args, "hidden_size", None)
        num_attention_heads = getattr(args, "num_attention_heads", None)
        head_dim = getattr(args, "head_dim", None)
        if head_dim is None and hidden_size is not None and num_attention_heads:
            head_dim = hidden_size // num_attention_heads
        if head_dim is not None and head_dim % 32 != 0:
            return CompatibilityResult(
                supported=False,
                reason=(
                    "adaptive_kv_v1 compressed tier requires llama head_dim divisible by 32"
                ),
            )

    baseline = model.make_cache()
    if not isinstance(baseline, list) or not baseline:
        return CompatibilityResult(
            supported=False,
            reason="adaptive_kv_v1 requires a non-empty per-layer cache list",
        )
    if any(type(layer) is not KVCache for layer in baseline):
        return CompatibilityResult(
            supported=False,
            reason="adaptive_kv_v1 requires a homogeneous list[KVCache] baseline",
        )
    return CompatibilityResult(supported=True, num_layers=len(baseline))
