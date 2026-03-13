"""Tests for model registry (§7.2, AC17).

Patterns: happy path, negative path, compatibility (new models
don't touch generate/cache/batch/server), alternate flows (aliases).
"""

from __future__ import annotations

import pytest

from mlxs.load.registry import _MODEL_REMAPPING, MODEL_REGISTRY, get_model_classes


class TestRegistryHappyPath:
    def test_llama_registered(self) -> None:
        model_cls, args_cls = get_model_classes("llama")
        assert model_cls.__name__ == "Model"
        assert args_cls.__name__ == "ModelArgs"

    def test_qwen2_registered(self) -> None:
        model_cls, args_cls = get_model_classes("qwen2")
        assert model_cls.__name__ == "Model"
        assert args_cls.__name__ == "ModelArgs"


class TestRegistryAliases:
    def test_mistral_resolves_to_llama(self) -> None:
        """Mistral uses Llama architecture (same weights format)."""
        model_cls, _ = get_model_classes("mistral")
        # Should resolve to llama module
        llama_cls, _ = get_model_classes("llama")
        assert model_cls is llama_cls

    def test_qwen3_resolves_to_qwen2(self) -> None:
        """Qwen3 dense uses Qwen2 architecture."""
        model_cls, _ = get_model_classes("qwen3")
        qwen2_cls, _ = get_model_classes("qwen2")
        assert model_cls is qwen2_cls


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
