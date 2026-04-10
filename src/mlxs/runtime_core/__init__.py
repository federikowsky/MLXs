"""Canonical Layer 1 / Performance Core surface.

This package is the sovereign runtime center introduced in Phase 1.
Exports are resolved lazily so contract-only imports do not force the
full MLX runtime until a runtime-bearing symbol is actually used.
"""

from __future__ import annotations

from mlxs.runtime_core.contracts import CoreFinishSignal, CoreStepResult

__all__ = [
    "CoreExecutionPolicy",
    "CoreFinishSignal",
    "CoreState",
    "CoreStepResult",
    "CoreTerminationPolicy",
    "run_greedy",
]


def __getattr__(name: str) -> object:
    if name == "run_greedy":
        from mlxs.runtime_core.greedy import run_greedy

        return run_greedy
    if name in {"CoreExecutionPolicy", "CoreTerminationPolicy"}:
        from mlxs.runtime_core.policy import CoreExecutionPolicy, CoreTerminationPolicy

        return {
            "CoreExecutionPolicy": CoreExecutionPolicy,
            "CoreTerminationPolicy": CoreTerminationPolicy,
        }[name]
    if name == "CoreState":
        from mlxs.runtime_core.state import CoreState

        return CoreState
    raise AttributeError(name)
