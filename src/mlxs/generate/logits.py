"""Logits recipe and processor utilities for generation.

The decode engine resolves generation behavior once into a ``StepRecipe``.
Batch/speculative paths still use the simpler processor factory helpers below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from mlxs.generate.sampling import apply_min_p, apply_top_k, apply_top_p, categorical_sampling

LogitsProcessor = Callable[[mx.array, mx.array], mx.array]


@dataclass(frozen=True, slots=True)
class StepRecipe:
    """Resolved device-step policy for one generation request."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    repetition_context_size: int
    emit_logprobs: bool
    top_logprobs: int

    @property
    def greedy(self) -> bool:
        # This preserves the existing sampler semantics: temp=0 is greedy and
        # ignores top-p / top-k / min-p settings.
        return self.temperature == 0.0

    @property
    def history_capacity(self) -> int:
        if self.repetition_penalty == 1.0:
            return 0
        return self.repetition_context_size

    @property
    def needs_history(self) -> bool:
        return self.history_capacity > 0

    @property
    def emit_top_logprobs(self) -> bool:
        return self.emit_logprobs and self.top_logprobs > 0

    @property
    def supports_compiled_step(self) -> bool:
        return self.greedy and not self.needs_history


def create_step_recipe(
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    repetition_context_size: int = 20,
    emit_logprobs: bool = False,
    top_logprobs: int = 0,
) -> StepRecipe:
    """Resolve generation options into one engine-native recipe."""
    return StepRecipe(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        emit_logprobs=emit_logprobs,
        top_logprobs=top_logprobs,
    )


def init_history_state(capacity: int) -> mx.array | None:
    """Create the fixed-capacity device-side history buffer."""
    if capacity <= 0:
        return None
    return mx.zeros((capacity,), dtype=mx.int32)


def append_history_state(
    history_tokens: mx.array | None,
    history_size: int,
    token: mx.array,
    *,
    capacity: int,
) -> tuple[mx.array | None, int]:
    """Append one generated token to the fixed-capacity history buffer."""
    if capacity <= 0:
        return None, 0

    if history_tokens is None:
        history_tokens = init_history_state(capacity)
        history_size = 0
    assert history_tokens is not None

    token_1d = token.reshape((1,))
    if history_size < capacity:
        updated = mx.concatenate(
            [
                history_tokens[:history_size],
                token_1d,
                history_tokens[history_size + 1 :],
            ]
        )
        return updated, history_size + 1

    updated = mx.concatenate([history_tokens[1:], token_1d])
    return updated, capacity


def apply_repetition_penalty(
    logits: mx.array,
    history_tokens: mx.array | None,
    history_size: int,
    *,
    penalty: float,
) -> mx.array:
    """Apply the existing sign-aware repetition penalty from device history."""
    if penalty == 1.0 or history_tokens is None or history_size <= 0:
        return logits

    recent = history_tokens[:history_size]
    selected = logits[:, recent]
    selected = mx.where(
        selected < 0,
        selected * penalty,
        selected / penalty,
    )
    logits[:, recent] = selected
    return logits


def sample_from_logprobs(
    logprobs: mx.array,
    *,
    recipe: StepRecipe,
) -> mx.array:
    """Sample a token id from logprobs according to the resolved recipe."""
    if recipe.greedy:
        return mx.argmax(logprobs, axis=-1)

    filtered = logprobs
    if 0 < recipe.top_p < 1.0:
        filtered = apply_top_p(filtered, recipe.top_p)
    if recipe.min_p > 0.0:
        filtered = apply_min_p(filtered, recipe.min_p)
    if recipe.top_k > 0:
        filtered = apply_top_k(filtered, recipe.top_k)
    return cast(mx.array, categorical_sampling(filtered, recipe.temperature))


def make_repetition_penalty(
    penalty: float,
    context_size: int = 20,
) -> LogitsProcessor:
    """Create a repetition penalty processor for logits-processor style call sites."""

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
    """Build logits processors for batch/speculative paths."""
    processors: list[LogitsProcessor] = []
    if repetition_penalty != 1.0:
        processors.append(make_repetition_penalty(repetition_penalty, repetition_context_size))
    return processors
