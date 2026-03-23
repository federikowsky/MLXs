from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mlxs.adaptive_kv.compatibility import (
    assess_generation_compatibility,
    select_generation_adapter,
)
from mlxs.adaptive_kv.runtime import SupportLevel
from mlxs.cache.kv import KVCache


class _SupportedModel:
    model_type = "llama"

    def make_cache(self) -> list[KVCache]:
        return [KVCache(), KVCache()]


class _WrongModelType:
    model_type = "qwen"

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]


class _MixedCacheModel:
    model_type = "llama"

    def make_cache(self) -> list[Any]:
        return [KVCache(), object()]


class _SlidingLlamaModel(_SupportedModel):
    args = SimpleNamespace(
        layer_types=["full_attention", "sliding_attention"],
        hidden_size=64,
        num_attention_heads=2,
        head_dim=32,
    )


class _NarrowHeadDimModel(_SupportedModel):
    args = SimpleNamespace(
        layer_types=["full_attention"],
        hidden_size=16,
        num_attention_heads=2,
        head_dim=8,
    )


def test_llama_kvcache_baseline_is_supported() -> None:
    result = assess_generation_compatibility(
        _SupportedModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is True
    assert result.num_layers == 2
    assert result.adapter_name == "llama"
    assert result.support_level is SupportLevel.FULL


def test_unsupported_model_type_is_rejected() -> None:
    result = assess_generation_compatibility(
        _WrongModelType(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is False
    assert "model_type='llama'" in (result.reason or "")
    assert result.support_level is SupportLevel.UNSUPPORTED


def test_mixed_cache_list_is_rejected() -> None:
    result = assess_generation_compatibility(
        _MixedCacheModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is False
    assert "homogeneous list[KVCache]" in (result.reason or "")


def test_compile_decode_is_rejected() -> None:
    result = assess_generation_compatibility(
        _SupportedModel(),
        cache=None,
        compile_decode=True,
        quantized_kv_start=0,
    )

    assert result.supported is False
    assert "compile_decode=True" in (result.reason or "")
    assert result.support_level is SupportLevel.PARTIAL


def test_legacy_quantized_flow_is_rejected() -> None:
    result = assess_generation_compatibility(
        _SupportedModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=4,
    )

    assert result.supported is False
    assert "quantized_kv_start" in (result.reason or "")


def test_sliding_llama_baseline_is_rejected() -> None:
    result = assess_generation_compatibility(
        _SlidingLlamaModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is False
    assert "full-attention llama baseline" in (result.reason or "")


def test_unsupported_head_dim_is_rejected() -> None:
    result = assess_generation_compatibility(
        _NarrowHeadDimModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is False
    assert "head_dim divisible by 32" in (result.reason or "")
    assert result.support_level is SupportLevel.PARTIAL


def test_generation_adapter_selection_returns_llama_adapter_for_supported_model() -> None:
    selection = select_generation_adapter(
        _SupportedModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert selection.adapter is not None
    assert selection.capabilities.adapter_name == "llama"
    assert selection.capabilities.supported is True
