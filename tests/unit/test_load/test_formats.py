"""Tests for weight format registry — detection and dispatch (§7.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlxs._errors import ModelLoadError
from mlxs._types import WeightFormat
from mlxs.config.schema import ModelConfig
from mlxs.load.formats import get_effective_format, load_model_and_tokenizer


class TestEffectiveFormat:
    """Resolve effective format from config and auto-detect."""

    def test_explicit_paro(self) -> None:
        cfg = ModelConfig(model_path="/x", weight_format=WeightFormat.PARO)
        assert get_effective_format(Path("/tmp"), cfg) == WeightFormat.PARO

    def test_explicit_safetensors(self) -> None:
        cfg = ModelConfig(model_path="/x", weight_format=WeightFormat.SAFETENSORS)
        assert get_effective_format(Path("/tmp"), cfg) == WeightFormat.SAFETENSORS

    def test_auto_fallback_safetensors_when_no_config(self) -> None:
        cfg = ModelConfig(model_path="/x", weight_format=WeightFormat.AUTO)
        # Path with no config.json -> safetensors
        assert get_effective_format(Path("/nonexistent"), cfg) == WeightFormat.SAFETENSORS

    def test_auto_detect_paro_when_quant_method_paro(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "quantization_config": {"quant_method": "paroquant"}})
        )
        cfg = ModelConfig(model_path=str(tmp_path), weight_format=WeightFormat.AUTO)
        assert get_effective_format(tmp_path, cfg) == WeightFormat.PARO

    def test_auto_detect_awq_when_quant_method_awq(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization_config": {"quant_method": "awq"}})
        )
        cfg = ModelConfig(model_path=str(tmp_path), weight_format=WeightFormat.AUTO)
        assert get_effective_format(tmp_path, cfg) == WeightFormat.AWQ

    def test_auto_detect_gptq_when_quant_method_gptq(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization_config": {"quant_method": "gptq"}})
        )
        cfg = ModelConfig(model_path=str(tmp_path), weight_format=WeightFormat.AUTO)
        assert get_effective_format(tmp_path, cfg) == WeightFormat.GPTQ


class TestLoadDispatch:
    """load_model_and_tokenizer dispatches to the right loader."""

    def test_awq_raises_not_implemented(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization_config": {"quant_method": "awq"}})
        )
        cfg = ModelConfig(model_path=str(tmp_path), weight_format=WeightFormat.AWQ)
        with pytest.raises(ModelLoadError, match="AWQ.*not implemented"):
            load_model_and_tokenizer(tmp_path, cfg)

    def test_gptq_raises_not_implemented(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization_config": {"quant_method": "gptq"}})
        )
        cfg = ModelConfig(model_path=str(tmp_path), weight_format=WeightFormat.GPTQ)
        with pytest.raises(ModelLoadError, match="GPTQ.*not implemented"):
            load_model_and_tokenizer(tmp_path, cfg)

    def test_paro_raises_if_not_installed_or_invalid_path(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "quantization_config": {"quant_method": "paroquant"}})
        )
        cfg = ModelConfig(model_path=str(tmp_path), weight_format=WeightFormat.PARO)
        with pytest.raises(ModelLoadError) as exc_info:
            load_model_and_tokenizer(tmp_path, cfg)
        # Either "paroquant" not installed or load failed (e.g. no weights)
        assert "paroquant" in str(exc_info.value) or "PARO" in str(exc_info.value) or "load" in str(exc_info.value).lower()


class TestParoDetection:
    """PARO detection uses only config.json (no paroquant import)."""

    def test_detect_paro_no_config_returns_false(self) -> None:
        from mlxs.load.paro import _detect_paro
        # No config.json -> load_config raises; _detect_paro catches and returns False
        assert _detect_paro(Path("/nonexistent")) is False

    def test_detect_paro_true_when_quant_method_paroquant(self, tmp_path: Path) -> None:
        from mlxs.load.paro import _detect_paro
        (tmp_path / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "paroquant"}})
        )
        assert _detect_paro(tmp_path) is True

    def test_detect_paro_false_when_other_quant_method(self, tmp_path: Path) -> None:
        from mlxs.load.paro import _detect_paro
        (tmp_path / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "awq"}})
        )
        assert _detect_paro(tmp_path) is False
