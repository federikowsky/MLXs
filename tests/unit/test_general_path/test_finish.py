"""Tests for Layer 2 finish-reason mapping."""

from __future__ import annotations

from mlxs._types import FinishReason
from mlxs.general_path.finish import map_core_finish_reason
from mlxs.runtime_core.contracts import CoreFinishSignal


def test_map_core_finish_reason_none() -> None:
    assert map_core_finish_reason(None) is None


def test_map_core_finish_reason_stop() -> None:
    assert map_core_finish_reason(CoreFinishSignal.STOP) == FinishReason.STOP


def test_map_core_finish_reason_length() -> None:
    assert map_core_finish_reason(CoreFinishSignal.LENGTH) == FinishReason.LENGTH
