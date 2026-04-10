"""Layer 4 configuration surface adapters."""

from __future__ import annotations

from typing import Any

from mlxs.config.cli import parse_argv
from mlxs.config.loader import resolve
from mlxs.config.schema import AppConfig


def parse_product_argv(argv: list[str] | None = None) -> tuple[str, AppConfig, str | None]:
    """Layer 4 wrapper around the user-facing CLI/config surface."""
    return parse_argv(argv)


def resolve_product_config(
    *,
    config_path: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Layer 4 wrapper for product-facing config resolution."""
    return resolve(config_path=config_path, cli_overrides=cli_overrides)


__all__ = ["AppConfig", "parse_product_argv", "resolve_product_config"]
