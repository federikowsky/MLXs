from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlxs.adaptive_kv.block_types import PressureState
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.manager import AdaptiveKVManager
from mlxs.observability.metrics import InMemoryMetrics


def test_pressure_state_transitions_to_soft_and_hard() -> None:
    metrics = InMemoryMetrics()
    manager = AdaptiveKVManager(
        AdaptiveKVConfig(enabled=True, soft_budget_bytes=1, hard_budget_bytes=2),
        num_layers=1,
        metrics=metrics,
    )

    manager.resident_bytes = lambda: 1  # type: ignore[method-assign]
    manager._compute_pressure_state()
    assert manager._pressure_state is PressureState.SOFT

    manager.resident_bytes = lambda: 2  # type: ignore[method-assign]
    manager._compute_pressure_state()
    assert manager._pressure_state is PressureState.HARD


def test_config_budget_validation_rejects_inverted_limits() -> None:
    with pytest.raises(ValidationError):
        AdaptiveKVConfig(enabled=True, soft_budget_bytes=10, hard_budget_bytes=5)
