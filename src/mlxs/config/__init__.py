"""Configuration module — layered config resolution (§8, NFR1, AC3).

Usage:
    from mlxs.config import resolve, AppConfig

    config = resolve()                           # defaults only
    config = resolve(config_path="config.yaml")  # file + defaults
"""

from mlxs.config.loader import resolve
from mlxs.config.schema import (
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
