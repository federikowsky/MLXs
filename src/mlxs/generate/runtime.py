"""Internal staged decode runtime helpers.

This module keeps the decode hot path explicit:
- build a device-resident tensor step
- wait at the boundary
- materialize host data
- apply mutation boundary
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import mlx.core as mx

from mlxs._types import TokenLogprobs, TopLogprob
from mlxs.cache.kv import KVCache
from mlxs.generate.compile import ForwardRuntime
from mlxs.generate.logits import LogitsProcessor
from mlxs.generate.profile import DecodeProfiler
from mlxs.generate.stop import StopCondition

_HEAVY_UNCOMPILED_SYNC_PROMPT_TOKENS = 1024


@dataclass(slots=True)
class DecodePlan:
    """Resolved once-per-request decode policy."""

    sampler: Callable[[mx.array], mx.array]
    stop: StopCondition
    decoder: Callable[[int], str]
    logits_processors: tuple[LogitsProcessor, ...]
    prompt_token_count: int
    emit_logprobs: bool
    top_logprobs: int
    clear_cache_interval: int
    quantized_kv_start: int
    kv_bits: int | None
    kv_group_size: int


@dataclass(slots=True)
class TensorStep:
    """One device-side decode step awaiting host materialization."""

    token: mx.array
    logprobs: mx.array | None = None
    top_indices: mx.array | None = None

    def sync_payload(self) -> tuple[mx.array, ...]:
        payload: list[mx.array] = [self.token]
        if self.logprobs is not None:
            payload.append(self.logprobs)
        if self.top_indices is not None:
            payload.append(self.top_indices)
        return tuple(payload)


@dataclass(slots=True)
class MaterializedStep:
    """Host-side data derived from a tensor step."""

    token_id: int
    text: str
    logprobs: TokenLogprobs | None


class BoundaryDriver(Protocol):
    """Sync/enqueue/wait contract for decode boundary implementations."""

    def dispatch(self, step: TensorStep, *, seed: bool = False) -> None: ...

    def wait(self, step: TensorStep, *, seed: bool = False) -> None: ...


@dataclass(slots=True)
class SyncBoundaryDriver:
    """Synchronous boundary implementation.

    Dispatch is intentionally a no-op in the sync milestone. This keeps the
    staged runtime shape async-ready without forcing the next-step sync to
    happen before the current token is yielded.
    """

    profiler: DecodeProfiler | None = None

    def dispatch(self, step: TensorStep, *, seed: bool = False) -> None:
        del step, seed

    def wait(self, step: TensorStep, *, seed: bool = False) -> None:
        t0 = time.perf_counter() if self.profiler is not None else 0.0
        mx.eval(*step.sync_payload())
        if self.profiler is None:
            return
        elapsed = time.perf_counter() - t0
        if seed:
            self.profiler.seed_sync_eval_s += elapsed
        else:
            self.profiler.sync_eval_s.append(elapsed)


@dataclass(slots=True)
class AsyncBoundaryDriver:
    """Async enqueue + explicit wait boundary implementation."""

    profiler: DecodeProfiler | None = None

    def dispatch(self, step: TensorStep, *, seed: bool = False) -> None:
        if seed:
            return
        t0 = time.perf_counter() if self.profiler is not None else 0.0
        async_eval = cast(Callable[..., Any], mx.async_eval)
        async_eval(*step.sync_payload())
        if self.profiler is not None:
            self.profiler.record_async_enqueue(
                duration_s=time.perf_counter() - t0,
                seed=seed,
            )

    def wait(self, step: TensorStep, *, seed: bool = False) -> None:
        t0 = time.perf_counter() if self.profiler is not None else 0.0
        mx.eval(*step.sync_payload())
        if self.profiler is None:
            return
        elapsed = time.perf_counter() - t0
        if seed:
            self.profiler.seed_sync_eval_s += elapsed
        else:
            self.profiler.sync_eval_s.append(elapsed)


def make_boundary_driver(
    *,
    plan: DecodePlan,
    forward_runtime: ForwardRuntime,
    profiler: DecodeProfiler | None,
) -> BoundaryDriver:
    """Select the internal decode boundary driver.

    Async stays the default steady-state boundary. The only automatic fallback
    is for heavier uncompiled decode, where Milestone 4 showed a local
    regression while compiled and smaller-prompt regimes remained healthy.
    The env gate still provides explicit force-sync / force-async control.
    """
    mode, reason = _resolve_boundary_policy(plan=plan, forward_runtime=forward_runtime)
    if mode == "async":
        if profiler is not None:
            profiler.set_boundary_mode("async", reason=reason)
        return AsyncBoundaryDriver(profiler)
    if profiler is not None:
        profiler.set_boundary_mode("sync", reason=reason)
    return SyncBoundaryDriver(profiler)


def build_seed_step(
    first_logits: mx.array,
    *,
    plan: DecodePlan,
    profiler: DecodeProfiler | None,
) -> TensorStep:
    """Prepare the initial decode step from prefill logits."""
    t0 = time.perf_counter() if profiler is not None else 0.0
    logprobs = first_logits - mx.logsumexp(first_logits, keepdims=True)
    token = plan.sampler(logprobs)
    if profiler is not None:
        profiler.seed_tensor_tail_s += time.perf_counter() - t0
    top_indices = _prepare_top_indices(logprobs, plan=plan, profiler=profiler)
    return TensorStep(
        token=token,
        logprobs=logprobs if plan.emit_logprobs else None,
        top_indices=top_indices,
    )


def build_next_step(
    input_token: mx.array,
    *,
    forward_runtime: ForwardRuntime,
    tokens_generated: list[int],
    plan: DecodePlan,
    profiler: DecodeProfiler | None,
    forward_step: int,
    generation_token: int,
) -> TensorStep:
    """Prepare the next decode step without materializing it on host."""
    next_logits = forward_runtime.forward(
        input_token[None],
        forward_step=forward_step,
        generation_token=generation_token,
    )
    if profiler is not None:
        tail_t0 = time.perf_counter()

    next_logits = next_logits[:, -1, :]
    if plan.logits_processors:
        if profiler is not None:
            profiler.record_logits_processors(token_count=len(tokens_generated))
        all_tokens = (
            mx.array(tokens_generated)
            if tokens_generated
            else mx.array([], dtype=mx.int32)
        )
        for processor in plan.logits_processors:
            next_logits = processor(all_tokens, next_logits)

    logprobs = next_logits - mx.logsumexp(next_logits, keepdims=True)
    token = plan.sampler(logprobs)
    if profiler is not None:
        profiler.post_forward_tensor_s.append(time.perf_counter() - tail_t0)

    top_indices = _prepare_top_indices(logprobs, plan=plan, profiler=profiler)
    return TensorStep(
        token=token,
        logprobs=logprobs if plan.emit_logprobs else None,
        top_indices=top_indices,
    )


def materialize_step(
    step: TensorStep,
    *,
    plan: DecodePlan,
    profiler: DecodeProfiler | None,
) -> MaterializedStep:
    """Convert a synchronized tensor step into host-visible data."""
    token_id = int(step.token.item())
    text = plan.decoder(token_id)

    token_logprobs: TokenLogprobs | None = None
    if step.logprobs is not None:
        if profiler is not None:
            profiler.logprobs_steps += 1
        token_lp = step.logprobs[0, token_id].item()
        top_logprobs: tuple[TopLogprob, ...] = ()
        if step.top_indices is not None:
            if profiler is not None:
                profiler.top_logprobs_steps += 1
            top_logprobs = tuple(
                TopLogprob(
                    token_id=idx_int,
                    token=plan.decoder(idx_int),
                    logprob=float(step.logprobs[0, idx_int].item()),
                )
                for idx_int in (int(idx.item()) for idx in step.top_indices)
            )
        token_logprobs = TokenLogprobs(
            token_logprob=token_lp,
            top_logprobs=top_logprobs,
        )

    return MaterializedStep(
        token_id=token_id,
        text=text,
        logprobs=token_logprobs,
    )


def apply_mutation_boundary(
    cache: list[KVCache],
    *,
    plan: DecodePlan,
    step_index: int,
    forward_runtime: ForwardRuntime,
    profiler: DecodeProfiler | None,
) -> None:
    """Apply cache cleanup and cache mutation between steps."""
    t0 = time.perf_counter() if profiler is not None else 0.0

    if plan.clear_cache_interval > 0 and step_index % plan.clear_cache_interval == 0:
        mx.clear_cache()
        if profiler is not None:
            profiler.record_cache_clear()

    if (
        plan.quantized_kv_start > 0
        and plan.kv_bits is not None
        and step_index == plan.quantized_kv_start
    ):
        from mlxs.cache import convert_to_quantized

        cache[:] = convert_to_quantized(
            cache,
            kv_bits=plan.kv_bits,
            kv_group_size=plan.kv_group_size,
        )
        if profiler is not None:
            profiler.record_cache_replacement(compiled_active=forward_runtime.compiled_active)
        forward_runtime.on_cache_replaced(cache)

    if profiler is not None:
        profiler.mutation_boundary_s.append(time.perf_counter() - t0)


def _prepare_top_indices(
    logprobs: mx.array,
    *,
    plan: DecodePlan,
    profiler: DecodeProfiler | None,
) -> mx.array | None:
    if not plan.emit_logprobs or plan.top_logprobs <= 0:
        return None

    t0 = time.perf_counter() if profiler is not None else 0.0
    row = logprobs[0]
    top_indices = mx.argpartition(row, kth=-plan.top_logprobs)[-plan.top_logprobs :]
    top_indices = top_indices[mx.argsort(row[top_indices])[::-1]]
    if profiler is not None:
        profiler.logprob_aux_tensor_s.append(time.perf_counter() - t0)
    return top_indices


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_boundary_policy(
    *,
    plan: DecodePlan,
    forward_runtime: ForwardRuntime,
) -> tuple[str, str]:
    if not hasattr(mx, "async_eval"):
        return "sync", "no_async_eval"

    env_value = os.getenv("MLXS_DECODE_ASYNC_EVAL")
    if env_value is not None:
        if _env_truthy("MLXS_DECODE_ASYNC_EVAL"):
            return "async", "forced_async_env"
        return "sync", "forced_sync_env"

    if (
        not forward_runtime.compiled_active
        and plan.prompt_token_count >= _HEAVY_UNCOMPILED_SYNC_PROMPT_TOKENS
    ):
        return "sync", "auto_sync_heavy_uncompiled"

    return "async", "auto_async_default"
