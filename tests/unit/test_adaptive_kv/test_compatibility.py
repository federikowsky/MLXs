from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mlxs.adaptive_kv.adapters import default_generation_adapter, generation_adapter_registry
from mlxs.adaptive_kv.compatibility import (
    assess_generation_compatibility,
    select_generation_adapter,
)
from mlxs.adaptive_kv.runtime import RuntimeFamily, SupportLevel
from mlxs.cache.arrays import ArraysCache
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache


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


class _SupportedQwen35FullAttentionModel:
    model_type = "qwen3_5"
    args = SimpleNamespace(
        full_attention_interval=1,
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
    )
    model = SimpleNamespace(
        layers=[SimpleNamespace(is_linear=False), SimpleNamespace(is_linear=False)]
    )

    def make_cache(self) -> list[KVCache]:
        return [KVCache(), KVCache()]


class _HybridQwen35Model:
    model_type = "qwen3_5"
    args = SimpleNamespace(
        full_attention_interval=4,
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
    )
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(is_linear=True),
            SimpleNamespace(is_linear=False),
        ]
    )

    def make_cache(self) -> list[Any]:
        return [ArraysCache(size=2), KVCache()]


class _SupportedMinistral3FullAttentionModel:
    model_type = "ministral3"
    args = SimpleNamespace(
        layer_types=["full_attention", "full_attention"],
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
        sliding_window=None,
    )
    model = SimpleNamespace(
        layers=[SimpleNamespace(use_sliding=False), SimpleNamespace(use_sliding=False)]
    )

    def make_cache(self) -> list[KVCache]:
        return [KVCache(), KVCache()]


class _SlidingMinistral3Model:
    model_type = "ministral3"
    args = SimpleNamespace(
        layer_types=["full_attention", "sliding_attention"],
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
        sliding_window=64,
    )
    model = SimpleNamespace(
        layers=[SimpleNamespace(use_sliding=False), SimpleNamespace(use_sliding=True)]
    )

    def make_cache(self) -> list[Any]:
        return [KVCache(), RotatingKVCache(max_size=64, keep=0)]


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
    assert result.runtime_family is RuntimeFamily.FULL_KV


def test_unsupported_model_type_is_rejected() -> None:
    result = assess_generation_compatibility(
        _WrongModelType(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is False
    assert "no registered exact adapter" in (result.reason or "")
    assert result.support_level is SupportLevel.UNSUPPORTED
    assert result.runtime_family is RuntimeFamily.UNKNOWN


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
    assert result.runtime_family is RuntimeFamily.WINDOWED_KV


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
    assert selection.runtime_family is RuntimeFamily.FULL_KV


def test_qwen35_full_attention_only_baseline_is_supported() -> None:
    result = assess_generation_compatibility(
        _SupportedQwen35FullAttentionModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is True
    assert result.num_layers == 2
    assert result.adapter_name == "qwen3_5"
    assert result.support_level is SupportLevel.FULL
    assert result.runtime_family is RuntimeFamily.FULL_KV


def test_standard_qwen35_hybrid_runtime_is_supported_in_family_c() -> None:
    result = assess_generation_compatibility(
        _HybridQwen35Model(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is True
    assert result.adapter_name == "qwen3_5"
    assert result.support_level is SupportLevel.FULL
    assert result.runtime_family is RuntimeFamily.HYBRID_STATE
    assert result.num_layers == 2


def test_generation_adapter_selection_returns_qwen35_adapter_for_supported_model() -> None:
    selection = select_generation_adapter(
        _SupportedQwen35FullAttentionModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert selection.adapter is not None
    assert selection.capabilities.adapter_name == "qwen3_5"
    assert selection.capabilities.supported is True
    assert selection.runtime_family is RuntimeFamily.FULL_KV


def test_ministral3_full_attention_only_baseline_is_supported() -> None:
    result = assess_generation_compatibility(
        _SupportedMinistral3FullAttentionModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is True
    assert result.num_layers == 2
    assert result.adapter_name == "ministral3"
    assert result.support_level is SupportLevel.FULL
    assert result.runtime_family is RuntimeFamily.FULL_KV


def test_standard_ministral3_sliding_runtime_is_supported_in_family_b() -> None:
    result = assess_generation_compatibility(
        _SlidingMinistral3Model(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert result.supported is True
    assert result.adapter_name == "ministral3"
    assert result.support_level is SupportLevel.FULL
    assert result.runtime_family is RuntimeFamily.WINDOWED_KV
    assert result.num_layers == 2


def test_generation_adapter_selection_returns_ministral3_adapter_for_supported_model() -> None:
    selection = select_generation_adapter(
        _SupportedMinistral3FullAttentionModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert selection.adapter is not None
    assert selection.capabilities.adapter_name == "ministral3"
    assert selection.capabilities.supported is True
    assert selection.runtime_family is RuntimeFamily.FULL_KV


def test_generation_adapter_selection_exposes_runtime_components() -> None:
    selection = select_generation_adapter(
        _SupportedModel(),
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
    )

    assert selection.adapter is not None
    assert selection.platform is selection.adapter
    assert selection.runtime_substrate is selection.family_bindings.runtime_substrate
    assert selection.replay_backend is selection.family_bindings.replay_backend
    assert selection.layer_runtime_type is selection.family_bindings.layer_runtime_type


def test_generation_adapter_registry_exposes_retained_default_platform() -> None:
    registry = generation_adapter_registry()

    assert registry.default_generation_adapter().name == "llama"
    assert default_generation_adapter().name == "llama"
    assert registry.resolve_generation_adapter(_SupportedModel()) is not None
    assert registry.resolve_generation_adapter(_SupportedQwen35FullAttentionModel()) is not None
    assert registry.resolve_generation_adapter(_HybridQwen35Model()) is not None
    assert (
        registry.resolve_generation_adapter(_SupportedMinistral3FullAttentionModel())
        is not None
    )
    assert registry.resolve_generation_adapter(_SlidingMinistral3Model()) is not None
    assert registry.resolve_generation_adapter(_WrongModelType()) is None
