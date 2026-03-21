"""Tests for model registry (§7.2, AC17).

Patterns: happy path, negative path, compatibility (new models
don't touch generate/cache/batch/server), alternate flows (aliases).
"""

from __future__ import annotations

import pytest

from mlxs._types import ModelMode
from mlxs.load.registry import (
    _MODEL_REMAPPING,
    MODEL_REGISTRY,
    get_model_capabilities,
    get_model_classes,
    get_model_entry,
)


class TestRegistryHappyPath:
    def test_llama_registered(self) -> None:
        model_cls, args_cls = get_model_classes("llama")
        assert model_cls.__name__ == "Model"
        assert args_cls.__name__ == "ModelArgs"

    def test_qwen2_registered(self) -> None:
        model_cls, args_cls = get_model_classes("qwen2")
        assert model_cls.__name__ == "Model"
        assert args_cls.__name__ == "ModelArgs"

    def test_ouro_registered(self) -> None:
        model_cls, args_cls = get_model_classes("ouro")
        assert model_cls.__name__ == "Model"
        assert args_cls.__name__ == "ModelArgs"

    def test_qwen3_entry_exposes_capabilities(self) -> None:
        entry = get_model_entry("qwen3")
        assert entry.model_type == "qwen3"
        assert entry.capabilities.supports_multimodal is True
        assert ModelMode.MULTIMODAL in entry.capabilities.supported_model_modes


class TestRegistryAliases:
    def test_mistral_resolves_to_llama(self) -> None:
        """Mistral uses Llama architecture (same weights format)."""
        model_cls, _ = get_model_classes("mistral")
        # Should resolve to llama module
        llama_cls, _ = get_model_classes("llama")
        assert model_cls is llama_cls

    def test_qwen3_resolves_to_its_unified_family_module(self) -> None:
        """qwen3 now uses its own unified family implementation."""
        model_cls, args_cls = get_model_classes("qwen3")
        assert model_cls.__module__ == "mlxs.models.qwen3"
        assert args_cls.__module__ == "mlxs.models.qwen3"

    def test_aliases_share_canonical_capabilities(self) -> None:
        assert get_model_capabilities("mistral") == get_model_capabilities("llama")

class TestRegistryNegativePath:
    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model_type"):
            get_model_classes("nonexistent_arch")

    def test_error_lists_supported_types(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            get_model_classes("bad_type")
        msg = str(exc_info.value)
        assert "llama" in msg
        assert "qwen2" in msg

    def test_unknown_capabilities_raise(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model_type"):
            get_model_capabilities("bad_type")

    @pytest.mark.parametrize(
        "legacy_model_type",
        ["qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5_vl", "lfm2_vl"],
    )
    def test_removed_legacy_model_types_raise(self, legacy_model_type: str) -> None:
        with pytest.raises(ValueError, match=f"Unsupported model_type '{legacy_model_type}'"):
            get_model_classes(legacy_model_type)


class TestRegistryExtensibility:
    """AC17: adding a model requires only models/ + registry."""

    def test_registry_is_dict(self) -> None:
        assert isinstance(MODEL_REGISTRY, dict)

    def test_remapping_is_dict(self) -> None:
        assert isinstance(_MODEL_REMAPPING, dict)

    def test_each_entry_has_three_parts(self) -> None:
        for model_type, entry in MODEL_REGISTRY.items():
            assert len(entry) == 3, f"{model_type}: expected (module, Model, Args)"
            module_path, cls_name, args_name = entry
            assert "." in module_path
            assert cls_name == "Model"
            assert args_name == "ModelArgs"

    def test_lazy_import_pattern(self) -> None:
        """Registry entries use string module paths (lazy import)."""
        for _, entry in MODEL_REGISTRY.items():
            module_path = entry[0]
            assert module_path.startswith("mlxs.models.")


class TestBaseModelArgsFromDict:
    """BaseModelArgs.from_dict filters unknown keys for forward compatibility."""

    def test_from_dict_basic(self) -> None:
        _, args_cls = get_model_classes("llama")
        args = args_cls.from_dict({"hidden_size": 2048, "num_hidden_layers": 16})
        assert args.hidden_size == 2048
        assert args.num_hidden_layers == 16

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Forward-compatible: extra keys from newer config.json are silently ignored."""
        _, args_cls = get_model_classes("llama")
        args = args_cls.from_dict(
            {
                "hidden_size": 1024,
                "future_feature_flag": True,
                "unknown_param": 42,
            }
        )
        assert args.hidden_size == 1024
        assert not hasattr(args, "future_feature_flag")

    def test_from_dict_qwen(self) -> None:
        _, args_cls = get_model_classes("qwen2")
        args = args_cls.from_dict(
            {
                "hidden_size": 2048,
                "vocab_size": 151936,
                "attention_bias": True,
            }
        )
        assert args.hidden_size == 2048
        assert args.vocab_size == 151936
        assert args.attention_bias is True
