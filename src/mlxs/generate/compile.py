"""Decode engine device-step backend and compile/rebind support."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from mlxs.generate.logits import (
    StepRecipe,
    append_history_state,
    apply_repetition_penalty,
    sample_from_logprobs,
)
from mlxs.generate.profile import DecodeProfiler
from mlxs.protocols.cache import CacheProtocol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeviceStep:
    """One device step's host-visible payload plus hidden next-step state."""

    token: mx.array
    token_logprob: mx.array | None = None
    top_token_ids: mx.array | None = None
    top_token_logprobs: mx.array | None = None
    history_tokens: mx.array | None = None
    history_size: int = 0

    def boundary_payload(self) -> tuple[mx.array, ...]:
        payload: list[mx.array] = [self.token]
        if self.token_logprob is not None:
            payload.append(self.token_logprob)
        if self.top_token_ids is not None:
            payload.append(self.top_token_ids)
        if self.top_token_logprobs is not None:
            payload.append(self.top_token_logprobs)
        return tuple(payload)


CompiledStepFn = Callable[[mx.array], mx.array | tuple[mx.array, ...]]


def make_compiled_step_backend(
    model: nn.Module,
    cache: list[CacheProtocol],
    recipe: StepRecipe,
) -> CompiledStepFn:
    """Compile the steady-state device step for recipe-supported requests."""
    if not recipe.supports_compiled_step:
        raise ValueError(
            "unsupported compiled recipe: compiled decode currently supports greedy/no-history"
        )

    if recipe.emit_top_logprobs:
        top_k = recipe.top_logprobs

        @mx.compile
        def compiled_step_with_top(
            input_ids: mx.array,
        ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
            logits = cast(mx.array, model(input_ids, cache=cache))[:, -1, :]
            token = mx.argmax(logits, axis=-1)
            lse = mx.logsumexp(logits, keepdims=True)
            token_logprob = mx.take_along_axis(logits, token[:, None], axis=-1).reshape(-1)
            token_logprob = token_logprob - lse.reshape(-1)
            row = logits[0]
            top_ids = mx.argpartition(row, kth=-top_k)[-top_k:]
            top_ids = top_ids[mx.argsort(row[top_ids])[::-1]]
            top_logprobs = row[top_ids] - lse[0, 0]
            return token, token_logprob, top_ids, top_logprobs

        return compiled_step_with_top

    if recipe.emit_logprobs:

        @mx.compile
        def compiled_step_with_logprobs(input_ids: mx.array) -> tuple[mx.array, mx.array]:
            logits = cast(mx.array, model(input_ids, cache=cache))[:, -1, :]
            token = mx.argmax(logits, axis=-1)
            lse = mx.logsumexp(logits, keepdims=True)
            token_logprob = mx.take_along_axis(logits, token[:, None], axis=-1).reshape(-1)
            token_logprob = token_logprob - lse.reshape(-1)
            return token, token_logprob

        return compiled_step_with_logprobs

    @mx.compile
    def compiled_step_greedy(input_ids: mx.array) -> mx.array:
        logits = cast(mx.array, model(input_ids, cache=cache))[:, -1, :]
        return mx.argmax(logits, axis=-1)

    return compiled_step_greedy


@dataclass(slots=True)
class StepBackend:
    """Own steady-state device-step execution and compile/cache lifecycle."""

    model: nn.Module
    cache: list[CacheProtocol]
    recipe: StepRecipe
    compile_requested: bool
    profiler: DecodeProfiler | None = None
    _compiled_step: CompiledStepFn | None = None
    _uncompiled_step: Callable[[mx.array], mx.array] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._uncompiled_step = partial(self.model, cache=self.cache)

    @classmethod
    def create(
        cls,
        model: nn.Module,
        cache: list[CacheProtocol],
        *,
        recipe: StepRecipe,
        compile_decode: bool,
        profiler: DecodeProfiler | None = None,
    ) -> StepBackend:
        backend = cls(
            model=model,
            cache=cache,
            recipe=recipe,
            compile_requested=compile_decode,
            profiler=profiler,
        )
        if compile_decode:
            backend._build_initial_compiled_step()
        return backend

    @property
    def compiled_active(self) -> bool:
        return self._compiled_step is not None

    def process_prefill_logits(
        self,
        logits: mx.array,
        *,
        history_tokens: mx.array | None,
        history_size: int,
    ) -> DeviceStep:
        return _process_logits_eager(
            logits,
            recipe=self.recipe,
            history_tokens=history_tokens,
            history_size=history_size,
        )

    def step_from_token(
        self,
        input_token: mx.array,
        *,
        history_tokens: mx.array | None,
        history_size: int,
        forward_step: int,
        generation_token: int,
    ) -> DeviceStep:
        step_t0 = time.perf_counter() if self.profiler is not None else 0.0
        if self._compiled_step is not None:
            compiled_output = self._compiled_step(input_token[None])
            if self.profiler is not None:
                self.profiler.record_forward_call(
                    duration_s=time.perf_counter() - step_t0,
                    using_compiled=True,
                    forward_step=forward_step,
                    generation_token=generation_token,
                )
            return _wrap_compiled_output(compiled_output, recipe=self.recipe)

        logits = self._uncompiled_step(input_token[None])
        if self.profiler is not None:
            self.profiler.record_forward_call(
                duration_s=time.perf_counter() - step_t0,
                using_compiled=False,
                forward_step=forward_step,
                generation_token=generation_token,
            )
            tail_t0 = time.perf_counter()
        if self.recipe.needs_history and self.profiler is not None:
            self.profiler.record_logits_processors(token_count=history_size)
        step = _process_logits_eager(
            logits[:, -1, :],
            recipe=self.recipe,
            history_tokens=history_tokens,
            history_size=history_size,
        )
        if self.profiler is not None:
            self.profiler.post_forward_tensor_s.append(time.perf_counter() - tail_t0)
        return step

    def on_cache_replaced(self, cache: list[CacheProtocol]) -> None:
        self.cache = cache
        self._uncompiled_step = partial(self.model, cache=self.cache)
        if not self.compile_requested or not self.compiled_active:
            return
        try:
            self._compiled_step = make_compiled_step_backend(self.model, self.cache, self.recipe)
        except Exception as exc:
            self._compiled_step = None
            if self.profiler is not None:
                self.profiler.record_compile_rebind(
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
        else:
            if self.profiler is not None:
                self.profiler.record_compile_rebind(success=True, error=None)

    def _build_initial_compiled_step(self) -> None:
        build_t0 = time.perf_counter()
        try:
            self._compiled_step = make_compiled_step_backend(self.model, self.cache, self.recipe)
        except Exception as exc:
            self._compiled_step = None
            if self.profiler is not None:
                self.profiler.record_compile_build(
                    duration_s=time.perf_counter() - build_t0,
                    available=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
        else:
            if self.profiler is not None:
                self.profiler.record_compile_build(
                    duration_s=time.perf_counter() - build_t0,
                    available=True,
                    error=None,
                )


def warmup(
    model: nn.Module,
    cache_factory: Callable[[], list[CacheProtocol]],
    *,
    vocab_size: int = 32000,
) -> None:
    """Run a dummy forward pass to trigger MLX compilation ahead of serving."""
    del vocab_size
    logger.info("Running warmup forward pass...")
    cache = cache_factory()
    dummy_input = mx.array([[0]])
    logits = model(dummy_input, cache=cache)
    mx.eval(logits)
    del cache, logits, dummy_input
    mx.clear_cache()
    logger.info("Warmup complete")


def _process_logits_eager(
    logits: mx.array,
    *,
    recipe: StepRecipe,
    history_tokens: mx.array | None,
    history_size: int,
) -> DeviceStep:
    if recipe.needs_history:
        logits = apply_repetition_penalty(
            logits,
            history_tokens,
            history_size,
            penalty=recipe.repetition_penalty,
        )

    if recipe.greedy:
        token = mx.argmax(logits, axis=-1)
        token_logprob: mx.array | None = None
        top_ids: mx.array | None = None
        top_logprobs: mx.array | None = None
        if recipe.emit_logprobs:
            lse = mx.logsumexp(logits, keepdims=True)
            token_logprob = mx.take_along_axis(logits, token[:, None], axis=-1).reshape(-1)
            token_logprob = token_logprob - lse.reshape(-1)
            if recipe.emit_top_logprobs:
                top_ids, top_logprobs = _top_logprob_payload(logits, lse, recipe.top_logprobs)
    else:
        base_logprobs = logits - mx.logsumexp(logits, keepdims=True)
        token = sample_from_logprobs(base_logprobs, recipe=recipe)
        token_logprob = None
        top_ids = None
        top_logprobs = None
        if recipe.emit_logprobs:
            token_logprob = mx.take_along_axis(base_logprobs, token[:, None], axis=-1).reshape(-1)
            if recipe.emit_top_logprobs:
                top_ids, top_logprobs = _top_logprob_payload(
                    base_logprobs,
                    None,
                    recipe.top_logprobs,
                    already_logprobs=True,
                )

    next_history_tokens, next_history_size = append_history_state(
        history_tokens,
        history_size,
        token,
        capacity=recipe.history_capacity,
    )
    return DeviceStep(
        token=token,
        token_logprob=token_logprob,
        top_token_ids=top_ids,
        top_token_logprobs=top_logprobs,
        history_tokens=next_history_tokens,
        history_size=next_history_size,
    )


def _top_logprob_payload(
    values: mx.array,
    lse: mx.array | None,
    top_k: int,
    *,
    already_logprobs: bool = False,
) -> tuple[mx.array, mx.array]:
    row = values[0]
    top_ids = mx.argpartition(row, kth=-top_k)[-top_k:]
    top_ids = top_ids[mx.argsort(row[top_ids])[::-1]]
    top_values = row[top_ids]
    if already_logprobs or lse is None:
        return top_ids, top_values
    return top_ids, top_values - lse[0, 0]


def _wrap_compiled_output(
    compiled_output: mx.array | tuple[mx.array, ...],
    *,
    recipe: StepRecipe,
) -> DeviceStep:
    if isinstance(compiled_output, tuple):
        if recipe.emit_top_logprobs:
            token, token_logprob, top_ids, top_logprobs = compiled_output
            return DeviceStep(
                token=token,
                token_logprob=token_logprob,
                top_token_ids=top_ids,
                top_token_logprobs=top_logprobs,
            )
        token, token_logprob = compiled_output
        return DeviceStep(
            token=token,
            token_logprob=token_logprob,
        )
    return DeviceStep(token=compiled_output)
