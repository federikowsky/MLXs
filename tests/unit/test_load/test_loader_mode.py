"""Tests for load_model with model_mode (§7.4, FR12)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pytest

from mlxs._errors import ModelLoadError
from mlxs._types import ModelMode
from mlxs.load.loader import load_model


def _make_model_dir(config: Mapping[str, object], *, with_weights: bool = False) -> Path:
    """Create a temp dir with config.json (and optional dummy weights)."""
    tmpdir = Path(tempfile.mkdtemp())
    (tmpdir / "config.json").write_text(json.dumps(config))
    if with_weights:
        # Create an empty safetensors file for format detection
        import safetensors.numpy
        safetensors.numpy.save_file({}, str(tmpdir / "model.safetensors"))
    return tmpdir


class TestModelModeResolution:
    def test_multimodal_without_vision_config_raises(self) -> None:
        """MULTIMODAL mode on a model without vision_config raises."""
        config = {
            "model_type": "qwen2",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 256,
        }
        model_dir = _make_model_dir(config)
        with pytest.raises(ModelLoadError, match="vision_config"):
            load_model(model_dir, model_mode=ModelMode.MULTIMODAL)

    def test_text_mode_loads_vl_model_as_text(self) -> None:
        """TEXT mode on a VL model loads without vision encoder (lazy)."""
        config = {
            "model_type": "qwen2_vl",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 256,
            "vision_config": {
                "depth": 2,
                "embed_dim": 64,
                "hidden_size": 64,
                "num_heads": 4,
            },
        }
        model_dir = _make_model_dir(config)
        model = load_model(model_dir, lazy=True, model_mode=ModelMode.TEXT)
        # Should not have vision_tower when in text mode
        assert not hasattr(model, "vision_tower")

    @pytest.mark.parametrize(
        ("model_type", "config"),
        [
            (
                "qwen2",
                {
                    "model_type": "qwen2",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "intermediate_size": 128,
                    "vocab_size": 256,
                    "vision_config": {
                        "depth": 2,
                        "embed_dim": 64,
                        "hidden_size": 64,
                        "num_heads": 4,
                    },
                },
            ),
            (
                "qwen3",
                {
                    "model_type": "qwen3",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "intermediate_size": 128,
                    "vocab_size": 256,
                    "vision_config": {
                        "depth": 2,
                        "embed_dim": 64,
                        "hidden_size": 64,
                        "num_heads": 4,
                    },
                },
            ),
            (
                "qwen3_moe",
                {
                    "model_type": "qwen3_moe",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "intermediate_size": 128,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "moe_intermediate_size": 64,
                    "vocab_size": 256,
                    "vision_config": {
                        "depth": 2,
                        "embed_dim": 64,
                        "hidden_size": 64,
                        "num_heads": 4,
                    },
                },
            ),
        ],
    )
    def test_auto_mode_resolves_multimodal_for_canonical_qwen_families(
        self,
        model_type: str,
        config: dict[str, object],
    ) -> None:
        """Canonical Qwen family keys become multimodal when vision_config is present."""
        model_dir = _make_model_dir(config)
        model = load_model(model_dir, lazy=True, model_mode=ModelMode.AUTO)

        assert model.model_type == model_type
        assert hasattr(model, "vision_tower")

    @pytest.mark.parametrize("model_type", ["qwen2_vl", "qwen3_vl", "qwen3_vl_moe"])
    def test_text_mode_loads_legacy_vl_keys_without_vision_encoder(
        self,
        model_type: str,
    ) -> None:
        """Legacy VL keys still resolve safely in text mode."""
        text_config: dict[str, object] = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 256,
        }
        if model_type != "qwen2_vl":
            text_config |= {
                "num_key_value_heads": 2,
                "head_dim": 16,
            }
        else:
            text_config["num_key_value_heads"] = 4
        if model_type == "qwen3_vl_moe":
            text_config |= {
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "moe_intermediate_size": 64,
            }

        config: dict[str, object] = {
            "model_type": model_type,
            "vision_config": {
                "depth": 2,
                "embed_dim": 64,
                "hidden_size": 64,
                "num_heads": 4,
            },
        }
        if model_type == "qwen3_vl_moe":
            config["text_config"] = text_config
        else:
            config |= text_config

        model_dir = _make_model_dir(config)
        model = load_model(model_dir, lazy=True, model_mode=ModelMode.TEXT)

        assert model.model_type == model_type
        assert not hasattr(model, "vision_tower")

    @pytest.mark.parametrize(
        ("model_type", "config"),
        [
            (
                "lfm2",
                {
                    "model_type": "lfm2",
                    "text_config": {
                        "model_type": "lfm2",
                        "vocab_size": 256,
                        "hidden_size": 64,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "max_position_embeddings": 128,
                        "norm_eps": 1e-6,
                        "conv_bias": False,
                        "conv_L_cache": 4,
                        "block_dim": 64,
                        "block_ff_dim": 128,
                        "block_multiple_of": 64,
                        "block_ffn_dim_multiplier": None,
                        "block_auto_adjust_ff_dim": True,
                        "rope_theta": 10000.0,
                        "layer_types": ["full_attention", "short_conv"],
                    },
                    "vision_config": {
                        "hidden_size": 16,
                        "intermediate_size": 32,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 4,
                        "num_channels": 3,
                        "image_size": 4,
                        "patch_size": 2,
                        "num_patches": 4,
                    },
                },
            ),
            (
                "mistral3",
                {
                    "model_type": "mistral3",
                    "text_config": {
                        "model_type": "ministral3",
                        "hidden_size": 64,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "intermediate_size": 128,
                        "rms_norm_eps": 1e-6,
                        "vocab_size": 256,
                        "max_position_embeddings": 512,
                        "layer_types": ["full_attention", "full_attention"],
                    },
                    "vision_config": {
                        "model_type": "pixtral",
                        "num_hidden_layers": 1,
                        "hidden_size": 16,
                        "head_dim": 4,
                        "intermediate_size": 32,
                        "num_attention_heads": 4,
                        "image_size": 4,
                        "patch_size": 2,
                        "projection_dim": 16,
                        "num_channels": 3,
                        "rms_norm_eps": 1e-5,
                        "rope_theta": 10000.0,
                    },
                },
            ),
            (
                "pixtral",
                {
                    "model_type": "pixtral",
                    "text_config": {
                        "model_type": "llama",
                        "hidden_size": 64,
                        "num_hidden_layers": 2,
                        "intermediate_size": 128,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-6,
                        "vocab_size": 256,
                        "rope_theta": 10000.0,
                        "tie_word_embeddings": False,
                    },
                    "vision_config": {
                        "model_type": "pixtral",
                        "num_hidden_layers": 1,
                        "hidden_size": 16,
                        "head_dim": 4,
                        "intermediate_size": 32,
                        "num_attention_heads": 4,
                        "image_size": 4,
                        "patch_size": 2,
                        "projection_dim": 16,
                        "num_channels": 3,
                        "rms_norm_eps": 1e-5,
                        "rope_theta": 10000.0,
                    },
                },
            ),
        ],
    )
    def test_auto_mode_resolves_multimodal_for_bucket2_families(
        self,
        model_type: str,
        config: dict[str, object],
    ) -> None:
        """Bucket 2 families touched in P4 still resolve to MULTIMODAL via load_model."""
        model_dir = _make_model_dir(config)
        model = load_model(model_dir, lazy=True, model_mode=ModelMode.AUTO)

        assert model.model_type == model_type
        assert hasattr(model, "vision_tower")

    def test_text_mode_loads_lfm2_vl_without_vision_encoder(self) -> None:
        """Legacy lfm2_vl key still resolves safely in text mode."""
        config: dict[str, object] = {
            "model_type": "lfm2_vl",
            "text_config": {
                "model_type": "lfm2",
                "vocab_size": 256,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 128,
                "norm_eps": 1e-6,
                "conv_bias": False,
                "conv_L_cache": 4,
                "block_dim": 64,
                "block_ff_dim": 128,
                "block_multiple_of": 64,
                "block_ffn_dim_multiplier": None,
                "block_auto_adjust_ff_dim": True,
                "rope_theta": 10000.0,
                "layer_types": ["full_attention", "short_conv"],
            },
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_channels": 3,
                "image_size": 4,
                "patch_size": 2,
                "num_patches": 4,
            },
        }
        model_dir = _make_model_dir(config)
        model = load_model(model_dir, lazy=True, model_mode=ModelMode.TEXT)

        assert model.model_type == "lfm2_vl"
        assert not hasattr(model, "vision_tower")

    def test_auto_mode_resolves_text_for_text_model(self) -> None:
        """AUTO mode on a text model resolves to TEXT."""
        config = {
            "model_type": "qwen2",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 256,
        }
        model_dir = _make_model_dir(config)
        # Should not raise — AUTO resolves to TEXT since no vision_config
        model = load_model(model_dir, lazy=True, model_mode=ModelMode.AUTO)
        assert model is not None

    def test_auto_mode_resolves_multimodal_for_vl_model(self) -> None:
        """AUTO mode on a VL model resolves to MULTIMODAL."""
        config = {
            "model_type": "qwen2_vl",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 256,
            "vision_config": {
                "depth": 2,
                "embed_dim": 64,
                "hidden_size": 64,
                "num_heads": 4,
            },
        }
        model_dir = _make_model_dir(config)
        model = load_model(model_dir, lazy=True, model_mode=ModelMode.AUTO)
        # AUTO + vision_config → MULTIMODAL → should have vision_tower
        assert hasattr(model, "vision_tower")

    def test_qwen3_5_moe_text_mode_ignores_deferred_vision_config(self) -> None:
        """TEXT mode remains available for qwen3_5_moe even if vision_config is present."""
        config: dict[str, object] = {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "intermediate_size": 128,
                "max_position_embeddings": 256,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 2,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64,
                "moe_intermediate_size": 64,
                "vocab_size": 256,
                "tie_word_embeddings": False,
            },
            "vision_config": {"hidden_size": 64},
        }
        model_dir = _make_model_dir(config)
        model = load_model(model_dir, lazy=True, model_mode=ModelMode.TEXT)

        assert model.model_type == "qwen3_5_moe"
        assert not hasattr(model, "vision_tower")

    def test_qwen3_5_moe_auto_mode_rejects_deferred_multimodal_config(self) -> None:
        """AUTO mode does not silently promote deferred qwen3_5_moe multimodal configs."""
        config: dict[str, object] = {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "intermediate_size": 128,
                "max_position_embeddings": 256,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 2,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64,
                "moe_intermediate_size": 64,
                "vocab_size": 256,
                "tie_word_embeddings": False,
            },
            "vision_config": {"hidden_size": 64},
        }
        model_dir = _make_model_dir(config)

        with pytest.raises(ModelLoadError, match="intentionally deferred"):
            load_model(model_dir, lazy=True, model_mode=ModelMode.AUTO)

    def test_missing_config_raises(self) -> None:
        """Missing config.json raises ModelLoadError."""
        tmpdir = Path(tempfile.mkdtemp())
        with pytest.raises(ModelLoadError):
            load_model(tmpdir, lazy=True)

    def test_unknown_model_type_raises(self) -> None:
        """Unknown model_type raises ModelLoadError."""
        config = {"model_type": "nonexistent_model_xyz"}
        model_dir = _make_model_dir(config)
        with pytest.raises(ModelLoadError):
            load_model(model_dir, lazy=True)
