"""Layered config resolution: defaults → YAML file → env vars → CLI (§8.1).

The ``resolve()`` function is the single entry point. It merges layers in
order and returns a frozen ``AppConfig``. No other module should read env
vars, CLI args, or config files directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from mlxs._errors import InvalidConfigError
from mlxs.config.schema import AppConfig

logger = logging.getLogger(__name__)

_ENV_PREFIX = "MLX_INFER_"


def resolve(
    *,
    config_path: str | Path | None = None,
    env: dict[str, str] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Resolve configuration by merging layers (§8.1).

    Resolution order (each layer overrides the previous):
        1. Code defaults (from ``AppConfig`` Pydantic defaults).
        2. YAML config file (if ``config_path`` provided or ``CONFIG_PATH`` env).
        3. Environment variables prefixed with ``MLX_INFER_``.
        4. CLI overrides (passed as flat dict).

    Args:
        config_path: Path to YAML config file. Overrides ``CONFIG_PATH`` env.
        env: Environment variables to use. Defaults to ``os.environ``.
        cli_overrides: Flat dict of CLI overrides (e.g. ``{"model.model_path": "..."}``).

    Returns:
        Frozen ``AppConfig`` instance.

    Raises:
        InvalidConfigError: On invalid config file, unknown keys (strict mode),
            or validation failure.
    """
    merged: dict[str, Any] = {}

    # Layer 2: YAML file
    file_path = _resolve_config_path(config_path, env or os.environ)
    if file_path is not None:
        file_data = _load_yaml(file_path)
        _deep_merge(merged, file_data)

    # Layer 3: env vars
    env_data = _parse_env(env or os.environ)
    _deep_merge(merged, env_data)

    # Layer 4: CLI overrides
    if cli_overrides:
        cli_data = _unflatten(cli_overrides)
        _deep_merge(merged, cli_data)

    # Build and validate
    # When strict_validation=False (default), ignore unknown keys (§8.1).
    # When True, _Frozen's extra="forbid" raises on unknown keys.
    strict = merged.get("strict_validation", False)
    try:
        config = AppConfig(**merged) if strict else _build_lenient(merged)
    except InvalidConfigError:
        raise
    except Exception as exc:
        raise InvalidConfigError(f"Configuration validation failed: {exc}") from exc

    logger.info(
        "Config resolved (file=%s, env_overrides=%d, cli_overrides=%d)",
        file_path or "none",
        len(env_data),
        len(cli_overrides) if cli_overrides else 0,
    )
    return config


def _resolve_config_path(
    explicit: str | Path | None,
    environ: dict[str, str],
) -> Path | None:
    """Determine config file path from explicit arg or CONFIG_PATH env."""
    if explicit is not None:
        path = Path(explicit)
    elif "CONFIG_PATH" in environ:
        path = Path(environ["CONFIG_PATH"])
    else:
        return None

    if not path.is_file():
        raise InvalidConfigError(f"Config file not found: {path}")
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML config file."""
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise InvalidConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise InvalidConfigError(f"Config file must be a YAML mapping, got {type(data).__name__}")
    return data


def _parse_env(environ: dict[str, str]) -> dict[str, Any]:
    """Extract MLX_INFER_* env vars into a nested dict.

    Convention: ``MLX_INFER_MODEL__MODEL_PATH=foo`` → ``{"model": {"model_path": "foo"}}``.
    Double underscore ``__`` separates sections; single underscore is part of the key.
    Values are kept as strings — Pydantic coerces types.
    """
    result: dict[str, Any] = {}
    for key, value in environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        stripped = key[len(_ENV_PREFIX) :].lower()
        parts = stripped.split("__")
        _set_nested(result, parts, value)
    return result


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert ``{"model.model_path": "x"}`` to ``{"model": {"model_path": "x"}}``."""
    result: dict[str, Any] = {}
    for dotted_key, value in flat.items():
        parts = dotted_key.split(".")
        _set_nested(result, parts, value)
    return result


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _build_lenient(merged: dict[str, Any]) -> AppConfig:
    """Build AppConfig ignoring unknown keys (strict_validation=False, §8.1).

    Strips unknown top-level and section-level keys before validation so that
    _Frozen's extra="forbid" does not reject them.
    """
    from mlxs.config.schema import (
        AdaptiveKVConfig,
        AppConfig,
        BatchConfig,
        CacheConfig,
        GenerateConfig,
        MemoryConfig,
        ModelConfig,
        ObservabilityConfig,
        PromptCacheConfig,
        ServerConfig,
        SpeculativeConfig,
        ToolCallingConfig,
    )

    section_models: dict[str, type] = {
        "model": ModelConfig,
        "generate": GenerateConfig,
        "memory": MemoryConfig,
        "cache": CacheConfig,
        "adaptive_kv": AdaptiveKVConfig,
        "prompt_cache": PromptCacheConfig,
        "batch": BatchConfig,
        "speculative": SpeculativeConfig,
        "tool_calling": ToolCallingConfig,
        "server": ServerConfig,
        "observability": ObservabilityConfig,
    }

    cleaned: dict[str, Any] = {}
    app_fields = set(AppConfig.model_fields)

    for key, val in merged.items():
        if key not in app_fields:
            logger.debug("Ignoring unknown config key: %s", key)
            continue
        if key in section_models and isinstance(val, dict):
            section_cls = section_models[key]
            known = set(section_cls.model_fields)
            cleaned[key] = {k: v for k, v in val.items() if k in known}
            dropped = set(val) - known
            for d in dropped:
                logger.debug("Ignoring unknown config key: %s.%s", key, d)
        else:
            cleaned[key] = val

    return AppConfig(**cleaned)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge ``override`` into ``base`` in-place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
