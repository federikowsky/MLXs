"""Decode engine runtime.

This replaces the old staged runtime/boundary-driver split with one engine that
owns prefill, steady-state decode, boundary policy, and cache mutation.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from mlxs._types import TokenEvent, TokenLogprobs, TopLogprob
from mlxs.generate.compile import DeviceStep, StepBackend
from mlxs.generate.logits import StepRecipe, init_history_state
from mlxs.generate.prefill import chunked_prefill
from mlxs.generate.profile import DecodeProfiler
from mlxs.generate.stop import StopMatcher
from mlxs.protocols.cache import CacheProtocol

_HEAVY_UNCOMPILED_SYNC_PROMPT_TOKENS = 1024
_generation_stream = mx.new_stream(mx.default_device())


@dataclass(slots=True)
class EnginePlan:
    """Resolved once-per-request host policy for the decode engine."""

    recipe: StepRecipe
    stop: StopMatcher
    decoder: Callable[[int], str]
    prompt_token_count: int
    clear_cache_interval: int
    quantized_kv_start: int
    kv_bits: int | None
    kv_group_size: int


@dataclass(slots=True)
class EngineState:
    """Mutable runtime state for one single-request generation."""

    cache: list[CacheProtocol]
    step_backend: StepBackend
    history_tokens: mx.array | None
    history_size: int
    boundary_mode: str
    boundary_policy_reason: str
    pending_step: DeviceStep | None = None
    generation_tokens: int = 0


class DecodeEngine:
    """Single-stream decode engine with one device-step contract."""

    __slots__ = ("_plan", "_profiler", "_state")

    def __init__(
        self,
        *,
        plan: EnginePlan,
        cache: list[CacheProtocol],
        step_backend: StepBackend,
        profiler: DecodeProfiler | None = None,
    ) -> None:
        mode, reason = _resolve_boundary_policy(
            prompt_token_count=plan.prompt_token_count,
            compiled_active=step_backend.compiled_active,
        )
        if profiler is not None:
            profiler.set_boundary_mode(mode, reason=reason)
        self._plan = plan
        self._state = EngineState(
            cache=cache,
            step_backend=step_backend,
            history_tokens=init_history_state(plan.recipe.history_capacity),
            history_size=0,
            boundary_mode=mode,
            boundary_policy_reason=reason,
        )
        self._profiler = profiler

    def run(
        self,
        prompt_tokens: mx.array,
        *,
        input_embeddings: mx.array | None = None,
        prefill_step_size: int = 2048,
    ) -> Iterator[TokenEvent]:
        """Run prefill + decode and yield TokenEvents lazily."""
        with mx.stream(_generation_stream):
            first_logits = chunked_prefill(
                self._state.step_backend.model,
                prompt_tokens,
                self._state.cache,
                prefill_step_size=prefill_step_size,
                input_embeddings=input_embeddings,
            )
            self._state.pending_step = self._build_seed_step(first_logits)
            try:
                yield from self._run_decode_loop()
            finally:
                if self._profiler is not None:
                    self._profiler.log_summary()

    def _run_decode_loop(self) -> Iterator[TokenEvent]:
        assert self._state.pending_step is not None
        current_step = self._state.pending_step
        self._dispatch(current_step, seed=True)

        while True:
            self._wait(current_step, seed=self._state.generation_tokens == 0)
            host_t0 = time.perf_counter() if self._profiler is not None else 0.0
            event = self._materialize_event(current_step)
            if self._profiler is not None:
                self._profiler.host_materialize_s.append(time.perf_counter() - host_t0)

            self._state.history_tokens = current_step.history_tokens
            self._state.history_size = current_step.history_size
            self._state.generation_tokens += 1

            if event.finish_reason is not None:
                yield event
                return

            self._apply_mutation_boundary(step_index=self._state.generation_tokens - 1)
            next_step = self._state.step_backend.step_from_token(
                current_step.token,
                history_tokens=self._state.history_tokens,
                history_size=self._state.history_size,
                forward_step=self._state.generation_tokens,
                generation_token=self._state.generation_tokens + 1,
            )
            self._dispatch(next_step)

            yield event
            current_step = next_step
            self._state.pending_step = current_step

    def _build_seed_step(self, first_logits: mx.array) -> DeviceStep:
        t0 = time.perf_counter() if self._profiler is not None else 0.0
        step = self._state.step_backend.process_prefill_logits(
            first_logits,
            history_tokens=self._state.history_tokens,
            history_size=self._state.history_size,
        )
        if self._profiler is not None:
            self._profiler.seed_tensor_tail_s += time.perf_counter() - t0
        return step

    def _dispatch(self, step: DeviceStep, *, seed: bool = False) -> None:
        if seed or self._state.boundary_mode == "sync":
            return
        t0 = time.perf_counter() if self._profiler is not None else 0.0
        async_eval = cast(Callable[..., object], mx.async_eval)
        async_eval(*step.boundary_payload())
        if self._profiler is not None:
            self._profiler.record_async_enqueue(
                duration_s=time.perf_counter() - t0,
                seed=seed,
            )

    def _wait(self, step: DeviceStep, *, seed: bool = False) -> None:
        t0 = time.perf_counter() if self._profiler is not None else 0.0
        mx.eval(*step.boundary_payload())
        if self._profiler is None:
            return
        elapsed = time.perf_counter() - t0
        if seed:
            self._profiler.seed_sync_eval_s += elapsed
        else:
            self._profiler.sync_eval_s.append(elapsed)

    def _materialize_event(self, step: DeviceStep) -> TokenEvent:
        token_id = int(step.token.item())
        text = self._plan.decoder(token_id)
        finish_reason = self._plan.stop.check(token_id, text)
        logprobs = self._materialize_logprobs(step)
        return TokenEvent(
            token_id=token_id,
            text=text,
            finish_reason=finish_reason,
            logprobs=logprobs,
            prompt_tokens=self._plan.prompt_token_count,
            generation_tokens=self._state.generation_tokens + 1,
        )

    def _materialize_logprobs(self, step: DeviceStep) -> TokenLogprobs | None:
        if step.token_logprob is None:
            return None
        if self._profiler is not None:
            self._profiler.logprobs_steps += 1

        top_logprobs: tuple[TopLogprob, ...] = ()
        if step.top_token_ids is not None and step.top_token_logprobs is not None:
            if self._profiler is not None:
                self._profiler.top_logprobs_steps += 1
            top_token_ids = [int(token_id.item()) for token_id in step.top_token_ids]
            top_token_logprobs = [
                float(token_logprob.item()) for token_logprob in step.top_token_logprobs
            ]
            top_logprobs = tuple(
                TopLogprob(
                    token_id=token_id,
                    token=self._plan.decoder(token_id),
                    logprob=token_logprob,
                )
                for token_id, token_logprob in zip(
                    top_token_ids,
                    top_token_logprobs,
                    strict=True,
                )
            )

        return TokenLogprobs(
            token_logprob=float(step.token_logprob.item()),
            top_logprobs=top_logprobs,
        )

    def _apply_mutation_boundary(self, *, step_index: int) -> None:
        t0 = time.perf_counter() if self._profiler is not None else 0.0

        if (
            self._plan.clear_cache_interval > 0
            and step_index % self._plan.clear_cache_interval == 0
        ):
            mx.clear_cache()
            if self._profiler is not None:
                self._profiler.record_cache_clear()

        if (
            self._plan.quantized_kv_start > 0
            and self._plan.kv_bits is not None
            and step_index == self._plan.quantized_kv_start
        ):
            from mlxs.cache import convert_to_quantized

            self._state.cache[:] = convert_to_quantized(
                self._state.cache,
                kv_bits=self._plan.kv_bits,
                kv_group_size=self._plan.kv_group_size,
            )
            if self._profiler is not None:
                self._profiler.record_cache_replacement(
                    compiled_active=self._state.step_backend.compiled_active
                )
            self._state.step_backend.on_cache_replaced(self._state.cache)
            self._refresh_boundary_policy()

        if self._profiler is not None:
            self._profiler.mutation_boundary_s.append(time.perf_counter() - t0)

    def _refresh_boundary_policy(self) -> None:
        mode, reason = _resolve_boundary_policy(
            prompt_token_count=self._plan.prompt_token_count,
            compiled_active=self._state.step_backend.compiled_active,
        )
        self._state.boundary_mode = mode
        self._state.boundary_policy_reason = reason
        if self._profiler is not None:
            self._profiler.set_boundary_mode(mode, reason=reason)


def run_decode_engine(
    prompt_tokens: mx.array,
    *,
    plan: EnginePlan,
    cache: list[CacheProtocol],
    step_backend: StepBackend,
    input_embeddings: mx.array | None = None,
    prefill_step_size: int = 2048,
    profiler: DecodeProfiler | None = None,
) -> Iterator[TokenEvent]:
    """Thin wrapper so callers do not need to instantiate the engine class."""
    engine = DecodeEngine(
        plan=plan,
        cache=cache,
        step_backend=step_backend,
        profiler=profiler,
    )
    return engine.run(
        prompt_tokens,
        input_embeddings=input_embeddings,
        prefill_step_size=prefill_step_size,
    )


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_boundary_policy(
    *,
    prompt_token_count: int,
    compiled_active: bool,
) -> tuple[str, str]:
    if not hasattr(mx, "async_eval"):
        return "sync", "no_async_eval"

    env_value = os.getenv("MLXS_DECODE_ASYNC_EVAL")
    if env_value is not None:
        if _env_truthy("MLXS_DECODE_ASYNC_EVAL"):
            return "async", "forced_async_env"
        return "sync", "forced_sync_env"

    if not compiled_active and prompt_token_count >= _HEAVY_UNCOMPILED_SYNC_PROMPT_TOKENS:
        return "sync", "auto_sync_heavy_uncompiled"

    return "async", "auto_async_default"
