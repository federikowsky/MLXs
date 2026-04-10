"""Layer 1 execution and termination policies."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlxs.runtime_core.contracts import CoreFinishSignal


@dataclass(frozen=True, slots=True)
class CoreTerminationPolicy:
    """Minimal termination policy for Layer 1."""

    max_tokens: int
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")

    def finish_for(
        self,
        token_id: int,
        *,
        generation_tokens: int,
    ) -> CoreFinishSignal | None:
        """Return the minimal finish signal after a token is produced."""
        if token_id in self.eos_token_ids:
            return CoreFinishSignal.STOP
        if generation_tokens >= self.max_tokens:
            return CoreFinishSignal.LENGTH
        return None


@dataclass(frozen=True, slots=True)
class CoreExecutionPolicy:
    """Execution-boundary controls owned by Layer 1."""

    clear_cache_interval: int = 0
    stream: Any | None = None

    def __post_init__(self) -> None:
        if self.clear_cache_interval < 0:
            raise ValueError("clear_cache_interval must be >= 0")


@contextmanager
def stream_context(policy: CoreExecutionPolicy) -> Iterator[None]:
    """Run work under the explicitly owned execution stream, if any."""
    if policy.stream is None:
        yield
        return
    with mx.stream(policy.stream):
        yield
