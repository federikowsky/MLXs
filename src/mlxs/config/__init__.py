"""Configuration module — Layer 4 config exposure surface."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AppConfig",
    "BatchConfig",
    "CacheConfig",
    "GenerateConfig",
    "MemoryConfig",
    "ModelConfig",
    "ObservabilityConfig",
    "PromptCacheConfig",
    "ServerConfig",
    "SpeculativeConfig",
    "ToolCallingConfig",
    "resolve",
]


def __getattr__(name: str) -> Any:
    if name == "resolve":
        from mlxs.config.loader import resolve

        return resolve
    if name in {
        "AppConfig",
        "BatchConfig",
        "CacheConfig",
        "GenerateConfig",
        "MemoryConfig",
        "ModelConfig",
        "ObservabilityConfig",
        "PromptCacheConfig",
        "ServerConfig",
        "SpeculativeConfig",
        "ToolCallingConfig",
    }:
        from mlxs.config import schema as _schema

        return getattr(_schema, name)
    raise AttributeError(name)
