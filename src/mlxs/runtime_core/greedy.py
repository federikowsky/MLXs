"""Canonical Phase 1 Layer 1 entrypoint."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import mlx.core as mx

from mlxs.protocols.model import ModelProtocol
from mlxs.runtime_core.contracts import CoreStepResult
from mlxs.runtime_core.decode import StepFn, decode_step
from mlxs.runtime_core.policy import CoreExecutionPolicy, CoreTerminationPolicy
from mlxs.runtime_core.prefill import run_prefill
from mlxs.runtime_core.selection import greedy_select
from mlxs.runtime_core.state import CoreState


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
    logits = run_prefill(
        model,
        prompt_tokens,
        active_state,
        execution=execution,
        prefill_step_size=prefill_step_size,
        input_embeddings=input_embeddings,
    )

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
