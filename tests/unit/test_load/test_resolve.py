"""Tests for model path resolution — local dir or Hugging Face id (FR1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mlxs._errors import ModelLoadError
from mlxs.load.resolve import resolve_model_path


class TestResolveLocalPath:
    """When path is an existing directory, return it unchanged."""

    def test_returns_same_path_for_directory(self, tmp_path: Path) -> None:
        assert resolve_model_path(tmp_path) == tmp_path
        assert resolve_model_path(str(tmp_path)) == tmp_path

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}")
        resolved = resolve_model_path(tmp_path)
        assert resolved.is_absolute()
        assert resolved == tmp_path.resolve()


class TestResolveHfId:
    """When path is not a directory, call snapshot_download (mocked)."""

    def test_calls_snapshot_download_and_returns_path(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snap:
            result = resolve_model_path("org/repo-id")
            snap.assert_called_once_with("org/repo-id")
            assert result == tmp_path

    def test_passes_revision_when_provided(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snap:
            resolve_model_path("org/repo", revision="v1.0")
            snap.assert_called_once_with("org/repo", revision="v1.0")

    def test_passes_token_when_provided(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snap:
            resolve_model_path("org/repo", token="hf_xxx")
            snap.assert_called_once_with("org/repo", token="hf_xxx")

    def test_passes_revision_and_token_when_both_provided(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snap:
            resolve_model_path("org/repo", revision="main", token="hf_yyy")
            snap.assert_called_once_with("org/repo", revision="main", token="hf_yyy")

    def test_does_not_pass_revision_or_token_when_none(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snap:
            resolve_model_path("org/repo")
            snap.assert_called_once_with("org/repo")


class TestResolveFailure:
    """Resolution failures raise ModelLoadError."""

    def test_invalid_hf_id_raises_model_load_error(self) -> None:
        with patch("huggingface_hub.snapshot_download") as snap:
            snap.side_effect = Exception("Repo not found")
            with pytest.raises(ModelLoadError, match="Failed to resolve model path") as exc_info:
                resolve_model_path("invalid/repo-id")
            assert "local directory or a valid Hugging Face" in str(exc_info.value)

    def test_network_error_raises_model_load_error(self) -> None:
        with patch("huggingface_hub.snapshot_download") as snap:
            snap.side_effect = ConnectionError("offline")
            with pytest.raises(ModelLoadError, match="Failed to resolve model path"):
                resolve_model_path("org/repo")
