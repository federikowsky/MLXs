"""Tests for load_model with model_mode (§7.4, FR12)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mlxs._errors import ModelLoadError
from mlxs._types import ModelMode
from mlxs.load.loader import load_model


def _make_model_dir(config: dict, *, with_weights: bool = False) -> Path:
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
