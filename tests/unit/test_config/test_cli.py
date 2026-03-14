"""Tests for config CLI — argv parsing and resolve integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlxs.config.schema import AppConfig


class TestParseArgvSubcommand:
    """parse_argv returns correct subcommand and calls resolve."""

    @patch("mlxs.config.cli.resolve")
    def test_serve_subcommand(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        sub, _, chat_query = parse_argv(["serve"])
        assert sub == "serve"
        assert chat_query is None
        mock_resolve.assert_called_once()
        call_kw = mock_resolve.call_args.kwargs
        assert call_kw["config_path"] is None
        assert (call_kw["cli_overrides"] or {}) == {}

    @patch("mlxs.config.cli.resolve")
    def test_chat_subcommand(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        sub, _, chat_query = parse_argv(["chat"])
        assert sub == "chat"
        assert chat_query is None

    @patch("mlxs.config.cli.resolve")
    def test_chat_query_is_returned(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        sub, _, chat_query = parse_argv(["chat", "write a haiku"])
        assert sub == "chat"
        assert chat_query == "write a haiku"

    @patch("mlxs.config.cli.resolve")
    def test_config_path_passed_to_resolve(self, mock_resolve: MagicMock, tmp_path: Path) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        cfg_path = tmp_path / "config.yaml"
        cfg_path.touch()
        parse_argv(["serve", "--config", str(cfg_path)])
        call_kw = mock_resolve.call_args.kwargs
        assert call_kw["config_path"] == cfg_path

    @patch("mlxs.config.cli.resolve")
    def test_cli_overrides_passed_to_resolve(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        parse_argv(["chat", "-o", "server.port=9090", "-o", "generate.max_tokens=256"])
        call_kw = mock_resolve.call_args.kwargs
        assert call_kw["cli_overrides"] == {"server.port": "9090", "generate.max_tokens": "256"}

    @patch("mlxs.config.cli.resolve")
    def test_explicit_flags_passed_to_resolve(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        parse_argv(["serve", "--port", "9090", "--host", "0.0.0.0", "--max-tokens", "128"])
        call_kw = mock_resolve.call_args.kwargs
        overrides = call_kw["cli_overrides"]
        assert overrides["server.port"] == 9090
        assert overrides["server.host"] == "0.0.0.0"
        assert overrides["generate.max_tokens"] == 128

    @patch("mlxs.config.cli.resolve")
    def test_explicit_flags_override_o(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        parse_argv(["serve", "-o", "server.port=8000", "--port", "9000"])
        call_kw = mock_resolve.call_args.kwargs
        assert call_kw["cli_overrides"]["server.port"] == 9000

    @patch("mlxs.config.cli.resolve")
    def test_key_equals_value_form(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        parse_argv(["serve", "--port=9090", "--host=0.0.0.0", "--max-tokens=256"])
        call_kw = mock_resolve.call_args.kwargs
        overrides = call_kw["cli_overrides"]
        assert overrides["server.port"] == 9090
        assert overrides["server.host"] == "0.0.0.0"
        assert overrides["generate.max_tokens"] == 256

    @patch("mlxs.config.cli.resolve")
    def test_no_flag_sets_false(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = AppConfig()
        from mlxs.config.cli import parse_argv

        parse_argv(["serve", "--no-metrics-enabled"])
        call_kw = mock_resolve.call_args.kwargs
        assert call_kw["cli_overrides"]["observability.metrics_enabled"] is False


class TestParseArgvValidation:
    """Missing subcommand or invalid args."""

    def test_no_subcommand_exits(self) -> None:
        from mlxs.config.cli import parse_argv

        with pytest.raises(SystemExit):
            parse_argv([])
