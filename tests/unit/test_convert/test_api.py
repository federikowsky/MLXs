from __future__ import annotations

from pathlib import Path

import pytest

from mlxs.convert.api import convert_source
from mlxs.convert.errors import ConverterError, UnsupportedRuntimeTargetError
from mlxs.convert.types import ConversionOptions, ConversionPhase


def test_convert_source_writes_failure_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DummyInspection:
        source_id = "fixture"
        warnings = ()

    def fake_inspect(*_args, **_kwargs):
        return DummyInspection()

    def fake_normalize(*_args, **_kwargs):
        raise UnsupportedRuntimeTargetError(
            "unsupported",
            phase=ConversionPhase.PLANNING,
        )

    monkeypatch.setattr("mlxs.convert.api._inspect_source", fake_inspect)
    monkeypatch.setattr("mlxs.convert.api.normalize_inspection", fake_normalize)

    with pytest.raises(UnsupportedRuntimeTargetError):
        convert_source("source", tmp_path, options=ConversionOptions())

    manifest_path = tmp_path / "conversion_manifest.json"
    assert manifest_path.is_file()
    manifest = manifest_path.read_text()
    assert "failure_phase" in manifest


def test_convert_source_wraps_raw_exceptions_into_converter_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DummyInspection:
        source_id = "fixture"
        warnings = ()
        tokenizer_artifacts = ()
        multimodal_artifacts = ()

    def fake_inspect(*_args, **_kwargs):
        return DummyInspection()

    def fake_normalize(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr("mlxs.convert.api._inspect_source", fake_inspect)
    monkeypatch.setattr("mlxs.convert.api.normalize_inspection", fake_normalize)

    with pytest.raises(ConverterError, match="Unexpected normalization failure: boom") as exc_info:
        convert_source("source", tmp_path, options=ConversionOptions())

    assert exc_info.value.phase == ConversionPhase.NORMALIZATION
    manifest = (tmp_path / "conversion_manifest.json").read_text()
    assert "failure_details" in manifest
    assert "ValueError" in manifest
