"""Canonical Phase 1 Layer 1 entrypoint."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import mlx.core as mx

from mlxs.protocols.model import ModelProtocol
from mlxs.runtime_core.contracts import CoreStepResult
from mlxs.runtime_core.decode import (
    StepFn,
    _schedule_next_decode_step_with_resolved_step,
    decode_step,
    materialize_prepared_step,
    prepare_decode_step,
)
from mlxs.runtime_core.policy import CoreExecutionPolicy, CoreTerminationPolicy
from mlxs.runtime_core.prefill import run_prefill
from mlxs.runtime_core.selection import greedy_select
from mlxs.runtime_core.state import CoreState

SHORT_PROMPT_LOOKAHEAD_THRESHOLD = 512


def run_greedy(
    model: ModelProtocol,
    prompt_tokens: Sequence[int] | mx.array,
    *,
    termination: CoreTerminationPolicy,
    execution: CoreExecutionPolicy,
    state: CoreState | None = None,
    prefill_step_size: int = 2048,
    input_embeddings: mx.array | None = None,
    step_fn: StepFn | None = None,
) -> Iterator[CoreStepResult]:
    """Run the canonical minimal Layer 1 path for a single request."""
    active_state = state if state is not None else CoreState.create(model)
    prompt_token_count = len(prompt_tokens)
    logits = run_prefill(
        model,
        prompt_tokens,
        active_state,
        execution=execution,
        prefill_step_size=prefill_step_size,
        input_embeddings=input_embeddings,
    )
    use_lookahead = prompt_token_count <= SHORT_PROMPT_LOOKAHEAD_THRESHOLD
    resolved_step_fn = (
        step_fn if step_fn is not None else (lambda input_ids: model(input_ids, cache=active_state.cache))
    )

    if use_lookahead:
        prepared = prepare_decode_step(
            logits,
            execution=execution,
            select_token=greedy_select,
            prime_token=True,
        )

        while True:
            next_prepared = None
            if active_state.generation_tokens + 1 < termination.max_tokens:
                next_prepared = _schedule_next_decode_step_with_resolved_step(
                    prepared,
                    execution=execution,
                    resolved_step=resolved_step_fn,
                    select_token=greedy_select,
                    prime_token=True,
                )

            result = materialize_prepared_step(
                active_state,
                prepared,
                termination=termination,
                execution=execution,
                next_prepared=next_prepared,
                force_eval=active_state.generation_tokens == 0,
            )
            yield result
            if result.finish is not None:
                return
            assert next_prepared is not None
            prepared = next_prepared
    else:
        while True:
            result, next_logits = decode_step(
                model,
                active_state,
                logits,
                termination=termination,
                execution=execution,
                select_token=greedy_select,
                step_fn=step_fn,
            )
            yield result
            if result.finish is not None:
                return
            assert next_logits is not None
            logits = next_logits
