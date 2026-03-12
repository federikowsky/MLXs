"""Shared test fixtures for MLXs.

Provides config factories, mock objects satisfying protocols, and common
test utilities. No MLX dependency — usable by unit tests without GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from mlxs.config.schema import (
    AppConfig,
    BatchConfig,
    CacheConfig,
    GenerateConfig,
    ModelConfig,
    ObservabilityConfig,
    PromptCacheConfig,
    ServerConfig,
)
from mlxs.observability.metrics import InMemoryMetrics, NoOpMetrics

# --- Config factories ---


@pytest.fixture
def default_config() -> AppConfig:
    """AppConfig with all defaults."""
    return AppConfig()


@pytest.fixture
def model_config() -> ModelConfig:
    """ModelConfig with all defaults."""
    return ModelConfig()


@pytest.fixture
def generate_config() -> GenerateConfig:
    """GenerateConfig with all defaults."""
    return GenerateConfig()


@pytest.fixture
def cache_config() -> CacheConfig:
    """CacheConfig with all defaults."""
    return CacheConfig()


@pytest.fixture
def prompt_cache_config() -> PromptCacheConfig:
    """PromptCacheConfig with all defaults."""
    return PromptCacheConfig()


@pytest.fixture
def batch_config() -> BatchConfig:
    """BatchConfig with all defaults."""
    return BatchConfig()


@pytest.fixture
def server_config() -> ServerConfig:
    """ServerConfig with all defaults."""
    return ServerConfig()


@pytest.fixture
def observability_config() -> ObservabilityConfig:
    """ObservabilityConfig with all defaults."""
    return ObservabilityConfig()


def make_config(**overrides: Any) -> AppConfig:
    """Create an AppConfig with selective overrides.

    Accepts top-level section overrides as dicts:
        make_config(model={"model_path": "/tmp/model"})
    """
    return AppConfig(**overrides)


# --- Metrics fixtures ---


@pytest.fixture
def noop_metrics() -> NoOpMetrics:
    """No-op metrics instance."""
    return NoOpMetrics()


@pytest.fixture
def memory_metrics() -> InMemoryMetrics:
    """In-memory metrics collector for assertions."""
    return InMemoryMetrics()
