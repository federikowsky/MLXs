"""Layer 2 finish-reason mapping above the minimal Layer 1 contract."""

from __future__ import annotations

from mlxs._types import FinishReason
from mlxs.runtime_core.contracts import CoreFinishSignal


def map_core_finish_reason(core_finish: CoreFinishSignal | None) -> FinishReason | None:
    """Map minimal Layer 1 finish signals to Layer 2 generation reasons."""
    if core_finish is None:
        return None
    if core_finish is CoreFinishSignal.STOP:
        return FinishReason.STOP
    if core_finish is CoreFinishSignal.LENGTH:
        return FinishReason.LENGTH
    raise ValueError(f"Unhandled core finish signal: {core_finish!r}")
