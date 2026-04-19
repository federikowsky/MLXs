"""Layer 1 decode-step runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class PreparedDecodeStep:
    """One Layer 1 token that has been selected and scheduled for materialization."""

    logits: mx.array
    token: mx.array


@dataclass(frozen=True, slots=True)
class PreparedNextLogits:
    """One raw next-logits handle produced by Layer 1 before Layer 2 transforms it."""

    logits: mx.array


def _default_step_fn(
    model: ModelProtocol,
    state: CoreState,
) -> StepFn:
    def step(input_ids: mx.array) -> mx.array:
        return model(input_ids, cache=state.cache)

    return step


def _resolve_step_fn(
    model: ModelProtocol,
    state: CoreState,
    step_fn: StepFn | None,
) -> StepFn:
    return step_fn if step_fn is not None else _default_step_fn(model, state)


def _prepare_next_logits_with_resolved_step(
    prepared: PreparedDecodeStep,
    *,
    execution: CoreExecutionPolicy,
    resolved_step: StepFn,
    prime_logits: bool = False,
) -> PreparedNextLogits:
    with stream_context(execution):
        next_logits = resolved_step(prepared.token[None])[:, -1, :]
        if prime_logits:
            mx.async_eval(next_logits)
    return PreparedNextLogits(logits=next_logits)


def _schedule_next_decode_step_with_resolved_step(
    prepared: PreparedDecodeStep,
    *,
    execution: CoreExecutionPolicy,
    resolved_step: StepFn,
    select_token: TokenSelector = greedy_select,
    prime_token: bool = True,
) -> PreparedDecodeStep:
    next_logits = _prepare_next_logits_with_resolved_step(
        prepared,
        execution=execution,
        resolved_step=resolved_step,
    ).logits
    with stream_context(execution):
        next_token = select_token(next_logits)
        if prime_token:
            mx.async_eval(next_token)
    return PreparedDecodeStep(logits=next_logits, token=next_token)


def _decode_step_with_resolved_step(
    state: CoreState,
    logits: mx.array,
    *,
    termination: CoreTerminationPolicy,
    execution: CoreExecutionPolicy,
    resolved_step: StepFn,
    select_token: TokenSelector = greedy_select,
) -> tuple[CoreStepResult, mx.array | None]:
    with stream_context(execution):
        token = select_token(logits)
        mx.eval(token)

    token_id = int(token.item())
    generation_tokens = state.increment_generation()
    finish = termination.finish_for(token_id, generation_tokens=generation_tokens)
    _clear_cache_if_due(execution, generation_tokens=generation_tokens)

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


def _clear_cache_if_due(
    execution: CoreExecutionPolicy,
    *,
    generation_tokens: int,
) -> None:
    if (
        execution.clear_cache_interval > 0
        and generation_tokens % execution.clear_cache_interval == 0
    ):
        mx.clear_cache()


def prepare_decode_step(
    logits: mx.array,
    *,
    execution: CoreExecutionPolicy,
    select_token: TokenSelector = greedy_select,
    prime_token: bool = True,
) -> PreparedDecodeStep:
    """Prepare the current token under Layer 1 stream discipline."""
    with stream_context(execution):
        token = select_token(logits)
        if prime_token:
            mx.async_eval(token)
    return PreparedDecodeStep(logits=logits, token=token)


def schedule_next_decode_step(
    model: ModelProtocol,
    state: CoreState,
    prepared: PreparedDecodeStep,
    *,
    execution: CoreExecutionPolicy,
    select_token: TokenSelector = greedy_select,
    step_fn: StepFn | None = None,
    prime_token: bool = True,
) -> PreparedDecodeStep:
    """Schedule the next token before the current one is materialized.

    This is the bounded Layer 1 lookahead seam used by the canonical
    greedy fast path when the caller can guarantee that a subsequent
    decode step is needed.
    """
    resolved_step = _resolve_step_fn(model, state, step_fn)
    return _schedule_next_decode_step_with_resolved_step(
        prepared,
        execution=execution,
        resolved_step=resolved_step,
        select_token=select_token,
        prime_token=prime_token,
    )


def prepare_next_logits(
    model: ModelProtocol,
    state: CoreState,
    prepared: PreparedDecodeStep,
    *,
    execution: CoreExecutionPolicy,
    step_fn: StepFn | None = None,
    prime_logits: bool = False,
) -> PreparedNextLogits:
    """Build raw next logits before Layer 2 transforms/selects the next token."""
    resolved_step = _resolve_step_fn(model, state, step_fn)
    return _prepare_next_logits_with_resolved_step(
        prepared,
        execution=execution,
        resolved_step=resolved_step,
        prime_logits=prime_logits,
    )


def materialize_prepared_step(
    state: CoreState,
    prepared: PreparedDecodeStep,
    *,
    termination: CoreTerminationPolicy,
    execution: CoreExecutionPolicy,
    next_prepared: PreparedDecodeStep | None = None,
    force_eval: bool = False,
) -> CoreStepResult:
    """Materialize a prepared token and emit the minimal Layer 1 result."""
    if next_prepared is not None:
        mx.eval(prepared.token, next_prepared.token)
    elif force_eval:
        mx.eval(prepared.token)

    token_id = int(prepared.token.item())
    generation_tokens = state.increment_generation()
    finish = termination.finish_for(token_id, generation_tokens=generation_tokens)
    _clear_cache_if_due(execution, generation_tokens=generation_tokens)
    return CoreStepResult(
        token_id=token_id,
        finish=finish,
        prompt_tokens=state.prompt_tokens,
        generation_tokens=generation_tokens,
    )


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
    resolved_step = _resolve_step_fn(model, state, step_fn)
    return _decode_step_with_resolved_step(
        state,
        logits,
        termination=termination,
        execution=execution,
        resolved_step=resolved_step,
        select_token=select_token,
    )
