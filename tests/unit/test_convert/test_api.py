from __future__ import annotations

from pathlib import Path

import pytest

from mlxs.convert.api import convert_source
from mlxs.convert.errors import UnsupportedRuntimeTargetError
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

    assert (tmp_path / "conversion_manifest.json").is_file()
