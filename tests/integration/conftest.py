"""Fixtures for integration tests."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests requiring MLX and optionally a model (deselect with '-m \"not integration\"')",
    )


@pytest.fixture(scope="session")
def e2e_model_path() -> str | None:
    """Path or HF id for e2e server/chat tests. None skips model-dependent tests."""
    return os.environ.get("MLXS_E2E_MODEL")
