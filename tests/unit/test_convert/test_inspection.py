from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlxs.convert.errors import UnsupportedSourceFormatError
from mlxs.convert.inspection import inspect_source


def test_inspect_source_rejects_non_safetensors_layout(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "pytorch_model.bin").write_bytes(b"bin")

    with pytest.raises(UnsupportedSourceFormatError):
        inspect_source(tmp_path)


def test_inspect_source_does_not_classify_tokenizer_files_as_multimodal(
    tmp_path: Path,
) -> None:
    from safetensors.numpy import save_file

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    save_file({"model.embed_tokens.weight": __import__("numpy").ones((2, 2), dtype="float32")}, str(tmp_path / "model.safetensors"))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")

    inspection = inspect_source(tmp_path)

    assert inspection.tokenizer_artifacts == ("tokenizer.json", "tokenizer_config.json")
    assert inspection.multimodal_artifacts == ()
