"""Layer 3 engine-local admission and backpressure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class AdmissionDecision(Enum):
    """Layer 3-internal admission result."""

    ADMIT = auto()
    DEFER = auto()
    REJECT = auto()


@dataclass(frozen=True, slots=True)
class EngineAdmissionPolicy:
    """Capacity-aware admission policy for engine-managed workloads.

    This policy is intentionally Layer 3-local. It does not define endpoint or
    transport semantics; Layer 4 may translate these outcomes later.
    """

    max_active: int | None = None
    max_pending: int | None = None

    def decide(
        self,
        *,
        active_count: int,
        pending_count: int,
    ) -> AdmissionDecision:
        if self.max_pending is not None and pending_count >= self.max_pending:
            return AdmissionDecision.REJECT
        if self.max_active is not None and active_count >= self.max_active:
            return AdmissionDecision.DEFER
        return AdmissionDecision.ADMIT
