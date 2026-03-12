"""Tests for config loader — layered resolution, env vars, YAML, CLI."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from mlxs._errors import InvalidConfigError
from mlxs.config.loader import resolve


class TestResolveDefaults:
    """resolve() with no arguments returns defaults."""

    def test_defaults_only(self) -> None:
        config = resolve()
        assert config.model.model_path == ""
        assert config.generate.temperature == 1.0
        assert config.strict_validation is False


class TestResolveYAML:
    """Layer 2: YAML file overrides defaults."""

    def test_yaml_overrides(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            dedent("""\
            model:
              model_path: /tmp/test-model
            generate:
              temperature: 0.7
              max_tokens: 1024
        """)
        )
        config = resolve(config_path=cfg_file)
        assert config.model.model_path == "/tmp/test-model"
        assert config.generate.temperature == 0.7
        assert config.generate.max_tokens == 1024
        # Non-overridden defaults preserved
        assert config.generate.top_p == 1.0

    def test_empty_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        config = resolve(config_path=cfg_file)
        assert config.model.model_path == ""

    def test_missing_file_raises(self) -> None:
        with pytest.raises(InvalidConfigError, match="not found"):
            resolve(config_path="/nonexistent/config.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("{{{{invalid yaml")
        with pytest.raises(InvalidConfigError, match="Invalid YAML"):
            resolve(config_path=cfg_file)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(InvalidConfigError, match="mapping"):
            resolve(config_path=cfg_file)


class TestResolveEnv:
    """Layer 3: env vars override file and defaults."""

    def test_env_override(self) -> None:
        env = {"MLX_INFER_MODEL__MODEL_PATH": "/env/model"}
        config = resolve(env=env)
        assert config.model.model_path == "/env/model"

    def test_env_overrides_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model:\n  model_path: /yaml/model\n")
        env = {"MLX_INFER_MODEL__MODEL_PATH": "/env/model"}
        config = resolve(config_path=cfg_file, env=env)
        assert config.model.model_path == "/env/model"

    def test_env_non_matching_ignored(self) -> None:
        env = {"OTHER_VAR": "value", "MLX_INFER_SERVER__PORT": "9090"}
        config = resolve(env=env)
        assert config.server.port == 9090

    def test_config_path_from_env(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model:\n  model_path: /from/env/path\n")
        env = {"CONFIG_PATH": str(cfg_file)}
        config = resolve(env=env)
        assert config.model.model_path == "/from/env/path"


class TestResolveCLI:
    """Layer 4: CLI overrides take highest precedence."""

    def test_cli_override(self) -> None:
        config = resolve(cli_overrides={"model.model_path": "/cli/model"})
        assert config.model.model_path == "/cli/model"

    def test_cli_overrides_env_and_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model:\n  model_path: /yaml\n")
        env = {"MLX_INFER_MODEL__MODEL_PATH": "/env"}
        config = resolve(
            config_path=cfg_file,
            env=env,
            cli_overrides={"model.model_path": "/cli"},
        )
        assert config.model.model_path == "/cli"

    def test_cli_nested_override(self) -> None:
        config = resolve(cli_overrides={"server.port": 3000})
        assert config.server.port == 3000


class TestResolveValidation:
    """Invalid merged config raises InvalidConfigError."""

    def test_invalid_merged_value(self) -> None:
        with pytest.raises(InvalidConfigError, match="validation failed"):
            resolve(cli_overrides={"generate.temperature": -5})
