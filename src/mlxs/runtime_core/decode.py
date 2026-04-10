"""Layer 1 decode-step runtime."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from mlxs.protocols.model import ModelProtocol
from mlxs.runtime_core.contracts import CoreStepResult
from mlxs.runtime_core.policy import (
    CoreExecutionPolicy,
    CoreTerminationPolicy,
    stream_context,
)
from mlxs.runtime_core.selection import TokenSelector, greedy_select
from mlxs.runtime_core.state import CoreState

StepFn = Callable[[mx.array], mx.array]


def _default_step_fn(
    model: ModelProtocol,
    state: CoreState,
) -> StepFn:
    def step(input_ids: mx.array) -> mx.array:
        return model(input_ids, cache=state.cache)

    return step


def decode_step(
    model: ModelProtocol,
    state: CoreState,
    logits: mx.array,
    *,
    termination: CoreTerminationPolicy,
    execution: CoreExecutionPolicy,
    select_token: TokenSelector = greedy_select,
    step_fn: StepFn | None = None,
) -> tuple[CoreStepResult, mx.array | None]:
    """Advance the decode state by exactly one token."""
    resolved_step = step_fn if step_fn is not None else _default_step_fn(model, state)

    with stream_context(execution):
        token = select_token(logits)
        mx.eval(token)

    token_id = int(token.item())
    generation_tokens = state.increment_generation()
    finish = termination.finish_for(token_id, generation_tokens=generation_tokens)

    if (
        execution.clear_cache_interval > 0
        and generation_tokens % execution.clear_cache_interval == 0
    ):
        mx.clear_cache()

    result = CoreStepResult(
        token_id=token_id,
        finish=finish,
        prompt_tokens=state.prompt_tokens,
        generation_tokens=generation_tokens,
    )
    if finish is not None:
        return result, None

    with stream_context(execution):
        next_logits = resolved_step(token[None])[:, -1, :]
    return result, next_logits
