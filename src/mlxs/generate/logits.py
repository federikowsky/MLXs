"""Logits processors — repetition penalty and custom processors (§6.1).

Processors are applied to logits before sampling. Each processor takes
(tokens, logits) and returns modified logits.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

LogitsProcessor = Callable[[mx.array, mx.array], mx.array]


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
    processors: list[LogitsProcessor] = []
    if repetition_penalty != 1.0:
        processors.append(make_repetition_penalty(repetition_penalty, repetition_context_size))
    return processors
