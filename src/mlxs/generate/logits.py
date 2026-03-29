"""Logits processors — repetition penalty and custom processors (§6.1).

Processors are applied to logits before sampling. Each processor takes
(tokens, logits) and returns modified logits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

LogitsProcessor = Callable[[mx.array, mx.array], mx.array]


@dataclass(frozen=True, slots=True)
class LogitsProcessorPlan:
    """Resolved logits processor plan for decode-time execution."""

    processors: tuple[LogitsProcessor, ...] = ()
    token_history_size: int = 0

    def apply(self, tokens: mx.array, logits: mx.array) -> mx.array:
        for processor in self.processors:
            logits = processor(tokens, logits)
        return logits

    @property
    def enabled(self) -> bool:
        return bool(self.processors)


def make_repetition_penalty(
    penalty: float,
    context_size: int = 20,
) -> LogitsProcessor:
    """Create a repetition penalty processor.

    Sign-aware multiplicative penalty: tokens with negative logits are
    multiplied by penalty, positive logits are divided by penalty.

    Args:
        penalty: Repetition penalty factor. 1.0 = no penalty.
        context_size: Number of recent tokens to consider.
    """

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if len(tokens) > 0:
            recent = tokens[-context_size:]
            selected = logits[:, recent]
            selected = mx.where(
                selected < 0,
                selected * penalty,
                selected / penalty,
            )
            logits[:, recent] = selected
        return logits

    return processor


def make_logits_processors(
    *,
    repetition_penalty: float = 1.0,
    repetition_context_size: int = 20,
) -> list[LogitsProcessor]:
    """Build a list of logits processors from generation options.

    Args:
        repetition_penalty: Multiplicative penalty for repeated tokens. 1.0 = disabled.
        repetition_context_size: Number of recent tokens to consider.

    Returns:
        List of processor functions (may be empty).
    """
    return list(
        make_logits_processor_plan(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        ).processors
    )


def make_logits_processor_plan(
    *,
    repetition_penalty: float = 1.0,
    repetition_context_size: int = 20,
) -> LogitsProcessorPlan:
    """Resolve the logits processor runtime plan once before decode."""

    processors: list[LogitsProcessor] = []
    token_history_size = 0
    if repetition_penalty != 1.0:
        processors.append(make_repetition_penalty(repetition_penalty, repetition_context_size))
        token_history_size = max(token_history_size, repetition_context_size)
    return LogitsProcessorPlan(tuple(processors), token_history_size)
