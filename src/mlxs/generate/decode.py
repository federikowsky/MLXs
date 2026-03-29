"""Decode runtime — staged single-request token generation.

The decode path is organized around four explicit phases:
- prepare once: resolve runtime plans and bounded state
- device step: forward, process logits, sample, and evaluate minimal tensors
- host materialization: decode text, compute stop reason, and build events
- mutation boundary: clear cache and handle cache/runtime replacement
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlxs._types import GenerateOptions, TokenEvent, TokenLogprobs, TopLogprob
from mlxs.cache import convert_to_quantized
from mlxs.generate.compile import DecodeForwardRuntime, make_decode_forward_runtime
from mlxs.generate.logits import LogitsProcessorPlan, make_logits_processor_plan
from mlxs.generate.sampling import SamplerFn, make_sampler
from mlxs.generate.stop import StopCondition

_EMPTY_TOKEN_HISTORY = mx.array([], dtype=mx.int32)


@dataclass(frozen=True, slots=True)
class DecodePlan:
    """Resolved single-request decode plan."""

    sampler: SamplerFn
    stop: StopCondition
    decoder: Callable[[int | list[int]], str]
    logits: LogitsProcessorPlan
    forward: DecodeForwardRuntime
    emit_logprobs: bool
    top_logprobs: int
    decode_text: bool
    prompt_token_count: int


@dataclass(slots=True)
class _PendingStep:
    token: mx.array
    token_logprob: mx.array | None = None
    top_token_ids: mx.array | None = None
    top_token_logprobs: mx.array | None = None


class _RecentTokenHistory:
    """Bounded recent-token history for logits processors."""

    __slots__ = ("_tokens",)

    def __init__(self, size: int) -> None:
        self._tokens: deque[int] | None = deque(maxlen=size) if size > 0 else None

    def append(self, token_id: int) -> None:
        if self._tokens is not None:
            self._tokens.append(token_id)

    def snapshot(self) -> mx.array:
        if not self._tokens:
            return _EMPTY_TOKEN_HISTORY
        return mx.array(tuple(self._tokens), dtype=mx.int32)


def prepare_decode_plan(
    model: Any,
    cache: list[Any],
    *,
    options: GenerateOptions,
    decoder: Callable[[int | list[int]], str],
    eos_token_id: int | None,
    prompt_token_count: int,
    compile_decode: bool = False,
) -> DecodePlan:
    """Resolve the staged decode runtime once before token generation."""

    return DecodePlan(
        sampler=make_sampler(
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            min_p=options.min_p,
        ),
        stop=StopCondition(
            eos_token_id=eos_token_id,
            max_tokens=options.max_tokens,
            stop_sequences=options.stop_sequences,
            extra_eos_token_ids=options.extra_eos_token_ids,
        ),
        decoder=decoder,
        logits=make_logits_processor_plan(repetition_penalty=options.repetition_penalty),
        forward=make_decode_forward_runtime(model, cache, compile_decode=compile_decode),
        emit_logprobs=options.logprobs,
        top_logprobs=options.top_logprobs,
        decode_text=True,
        prompt_token_count=prompt_token_count,
    )


def _sample_from_logits(
    logits: mx.array,
    *,
    plan: DecodePlan,
    history: _RecentTokenHistory,
) -> _PendingStep:
    if plan.logits.enabled:
        logits = plan.logits.apply(history.snapshot(), logits)

    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    token = plan.sampler(logprobs)

    pending = _PendingStep(token=token)
    to_eval: list[mx.array] = [token]

    if plan.emit_logprobs:
        token_logprob = mx.take_along_axis(logprobs, token[:, None], axis=-1)
        pending.token_logprob = token_logprob
        to_eval.append(token_logprob)

        if plan.top_logprobs > 0:
            top_count = min(plan.top_logprobs, logprobs.shape[-1])
            kth = logprobs.shape[-1] - top_count
            top_token_ids = mx.argpartition(logprobs, kth=kth, axis=-1)[:, -top_count:]
            top_token_logprobs = mx.take_along_axis(logprobs, top_token_ids, axis=-1)
            order = mx.argsort(top_token_logprobs, axis=-1)[:, ::-1]
            pending.top_token_ids = mx.take_along_axis(top_token_ids, order, axis=-1)
            pending.top_token_logprobs = mx.take_along_axis(top_token_logprobs, order, axis=-1)
            to_eval.extend((pending.top_token_ids, pending.top_token_logprobs))

    mx.eval(*to_eval)
    return pending


def _build_logprob_payload(
    pending: _PendingStep,
    *,
    decoder: Callable[[int | list[int]], str],
) -> TokenLogprobs | None:
    if pending.token_logprob is None:
        return None

    top_logprobs: tuple[TopLogprob, ...] = ()
    if pending.top_token_ids is not None and pending.top_token_logprobs is not None:
        entries: list[TopLogprob] = []
        for idx, logprob in zip(
            pending.top_token_ids[0], pending.top_token_logprobs[0], strict=False
        ):
            token_id = int(idx.item())
            entries.append(
                TopLogprob(
                    token_id=token_id,
                    token=decoder(token_id),
                    logprob=float(logprob.item()),
                )
            )
        top_logprobs = tuple(entries)

    return TokenLogprobs(
        token_logprob=float(pending.token_logprob.item()),
        top_logprobs=top_logprobs,
    )


def _materialize_event(
    pending: _PendingStep,
    *,
    plan: DecodePlan,
    generation_tokens: int,
) -> tuple[TokenEvent, int]:
    token_id = int(pending.token.item())
    text = plan.decoder(token_id) if plan.decode_text else ""

    finish_reason = plan.stop.check_token(token_id)
    if finish_reason is None and plan.stop.needs_text:
        finish_reason = plan.stop.check_text(text)

    event = TokenEvent(
        token_id=token_id,
        text=text,
        finish_reason=finish_reason,
        logprobs=_build_logprob_payload(pending, decoder=plan.decoder),
        prompt_tokens=plan.prompt_token_count,
        generation_tokens=generation_tokens,
    )
    return event, token_id


def _apply_mutation_boundary(
    *,
    step_index: int,
    cache: list[Any],
    forward: DecodeForwardRuntime,
    clear_cache_interval: int,
    quantized_kv_start: int,
    kv_bits: int | None,
    kv_group_size: int,
) -> None:
    if clear_cache_interval > 0 and step_index % clear_cache_interval == 0:
        mx.clear_cache()

    if quantized_kv_start > 0 and kv_bits is not None and step_index == quantized_kv_start:
        cache[:] = convert_to_quantized(cache, kv_bits=kv_bits, kv_group_size=kv_group_size)
        forward.on_cache_replaced(cache)


def decode_loop(
    cache: list[Any],
    first_logits: mx.array,
    *,
    plan: DecodePlan,
    clear_cache_interval: int = 256,
    quantized_kv_start: int = 0,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
) -> Iterator[TokenEvent]:
    """Run the staged decode loop, yielding one TokenEvent per generated token."""

    history = _RecentTokenHistory(plan.logits.token_history_size)
    pending = _sample_from_logits(first_logits, plan=plan, history=history)

    step_index = 0
    while True:
        event, token_id = _materialize_event(
            pending,
            plan=plan,
            generation_tokens=step_index + 1,
        )
        history.append(token_id)
        yield event

        if event.finish_reason is not None:
            return

        _apply_mutation_boundary(
            step_index=step_index,
            cache=cache,
            forward=plan.forward,
            clear_cache_interval=clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )

        step_index += 1
        next_logits = plan.forward.forward(mx.reshape(pending.token, (1, 1)))
        pending = _sample_from_logits(next_logits[:, -1, :], plan=plan, history=history)
