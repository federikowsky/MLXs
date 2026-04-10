"""Canonical Layer 1 / Performance Core surface.

This package is the sovereign runtime center introduced in Phase 1.
Legacy generation surfaces remain outside this package and are not
the canonical Layer 1 boundary.
"""

from mlxs.runtime_core.contracts import CoreFinishSignal, CoreStepResult
from mlxs.runtime_core.greedy import run_greedy
from mlxs.runtime_core.policy import CoreExecutionPolicy, CoreTerminationPolicy
from mlxs.runtime_core.state import CoreState

__all__ = [
    "CoreExecutionPolicy",
    "CoreFinishSignal",
    "CoreState",
    "CoreStepResult",
    "CoreTerminationPolicy",
    "run_greedy",
]
