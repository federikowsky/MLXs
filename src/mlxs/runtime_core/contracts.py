"""Layer 1 data contracts.

These contracts are intentionally smaller than the legacy generation
contracts. They carry only the data needed for token advancement and
minimal termination signaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class CoreFinishSignal(Enum):
    """Minimal terminal signals exposed by Layer 1."""

    STOP = auto()
    LENGTH = auto()


@dataclass(frozen=True, slots=True)
class CoreStepResult:
    """One token advancement result produced by Layer 1."""

    token_id: int
    finish: CoreFinishSignal | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
