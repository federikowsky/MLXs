"""Decode loop orchestration for single-request generation."""

from __future__ import annotations

import time
from collections.abc import Iterator

import mlx.core as mx

from mlxs._types import TokenEvent
from mlxs.generate.compile import ForwardRuntime
from mlxs.generate.profile import DecodeProfiler
from mlxs.generate.runtime import (
    DecodePlan,
    apply_mutation_boundary,
    build_next_step,
    build_seed_step,
    make_boundary_driver,
    materialize_step,
)


def decode_loop(
    first_logits: mx.array,
    *,
    plan: DecodePlan,
    forward_runtime: ForwardRuntime,
    profiler: DecodeProfiler | None = None,
) -> Iterator[TokenEvent]:
    """Run the staged decode loop, yielding one ``TokenEvent`` per token."""
    boundary = make_boundary_driver(profiler)
    tokens_generated: list[int] = []
    current_step = build_seed_step(first_logits, plan=plan, profiler=profiler)
    boundary.dispatch(current_step, seed=True)

    step_index = 0
    try:
        while True:
            boundary.wait(current_step, seed=step_index == 0)
            host_t0 = time.perf_counter() if profiler is not None else 0.0
            materialized = materialize_step(current_step, plan=plan, profiler=profiler)
            tokens_generated.append(materialized.token_id)
            finish_reason = plan.stop.check(materialized.token_id, materialized.text)

            event = TokenEvent(
                token_id=materialized.token_id,
                text=materialized.text,
                finish_reason=finish_reason,
                logprobs=materialized.logprobs,
                prompt_tokens=plan.prompt_token_count,
                generation_tokens=step_index + 1,
            )
            if profiler is not None:
                profiler.host_materialize_s.append(time.perf_counter() - host_t0)

            if finish_reason is not None:
                yield event
                return

            apply_mutation_boundary(
                forward_runtime.cache,
                plan=plan,
                step_index=step_index,
                forward_runtime=forward_runtime,
                profiler=profiler,
            )
            next_step = build_next_step(
                current_step.token,
                forward_runtime=forward_runtime,
                tokens_generated=tokens_generated,
                plan=plan,
                profiler=profiler,
                forward_step=step_index + 1,
                generation_token=step_index + 2,
            )
            boundary.dispatch(next_step)

            yield event
            current_step = next_step
            step_index += 1
    finally:
        if profiler is not None:
            profiler.log_summary()
