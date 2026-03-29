"""Decode runtime — staged single-request token generation.

The decode path is organized around four explicit phases:
- prepare once: resolve runtime plans and bounded state
- device step: forward, process logits, sample, and evaluate minimal tensors
- host materialization: decode text, compute stop reason, and build events
- mutation boundary: clear cache and handle cache/runtime replacement
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx

from mlxs._types import FinishReason, GenerateOptions, TokenEvent, TokenLogprobs, TopLogprob
from mlxs.cache import convert_to_quantized
from mlxs.generate.compile import DecodeForwardRuntime, make_decode_forward_runtime
from mlxs.generate.logits import LogitsProcessorPlan, make_logits_processor_plan
from mlxs.generate.sampling import SamplerFn, make_sampler
from mlxs.generate.stop import StopCondition

_EMPTY_TOKEN_HISTORY = mx.array([], dtype=mx.int32)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def decode_profile_enabled() -> bool:
    """True when ``MLXS_DECODE_PROFILE`` is set (decode timing to stderr)."""

    return _env_flag("MLXS_DECODE_PROFILE")


def decode_async_eval_enabled() -> bool:
    """True when ``MLXS_DECODE_ASYNC_EVAL`` is set (experimental overlap path)."""

    return _env_flag("MLXS_DECODE_ASYNC_EVAL")


def emit_decode_profile_report(
    profile: dict[str, Any],
    *,
    compile_decode: bool,
    async_eval: bool,
) -> None:
    """Emit one-line summary + breakdown to stderr (dev-only)."""

    n_fwd = int(profile.get("n_forward_decode", 0))
    lines = [
        "[MLXS_DECODE_PROFILE] NOTE: MLX defers work to sync boundaries. "
        "With async_eval disabled, mx_eval_s is synchronous drain time. "
        "With async_eval enabled, mx_async_eval_s is enqueue time and "
        "mx_eval_s is explicit wait time before host reads.",
        f"[MLXS_DECODE_PROFILE] compile_decode={compile_decode} "
        f"async_eval={async_eval} "
        f"forward_steps={n_fwd} "
        f"forward_decode_s={profile.get('forward_decode_s', 0.0):.6f} "
        f"logits_sample_prep_s={profile.get('logits_sample_prep_s', 0.0):.6f} "
        f"mx_async_eval_s={profile.get('mx_async_eval_s', 0.0):.6f} "
        f"mx_eval_s={profile.get('mx_eval_s', 0.0):.6f} "
        f"materialize_s={profile.get('materialize_s', 0.0):.6f} "
        f"mutation_s={profile.get('mutation_s', 0.0):.6f}",
    ]
    samples: list[float] = profile.get("forward_wall_samples") or []
    if samples:
        lines.append(
            f"[MLXS_DECODE_PROFILE] forward() wall only per decode step: "
            f"first={samples[0] * 1e3:.3f}ms "
            f"median={statistics.median(samples) * 1e3:.3f}ms "
            f"last={samples[-1] * 1e3:.3f}ms "
            f"max={max(samples) * 1e3:.3f}ms "
            f"min={min(samples) * 1e3:.3f}ms"
        )
    step_tot: list[float] = profile.get("step_wall_samples") or []
    if step_tot:
        label = (
            "decode dispatch (fwd+logits_prep+async_eval enqueue)"
            if async_eval
            else "full decode step (fwd+logits_prep+mx_eval)"
        )
        lines.append(
            f"[MLXS_DECODE_PROFILE] {label}: "
            f"median={statistics.median(step_tot) * 1e3:.3f}ms "
            f"sum={sum(step_tot):.4f}s over {len(step_tot)} steps"
        )
    total = sum(
        float(profile.get(k, 0.0))
        for k in (
            "forward_decode_s",
            "logits_sample_prep_s",
            "mx_async_eval_s",
            "mx_eval_s",
            "materialize_s",
            "mutation_s",
        )
    )
    if total > 0:
        lines.append(
            "[MLXS_DECODE_PROFILE] fraction of profiled wall time: "
            f"fwd={profile.get('forward_decode_s', 0) / total:.3f} "
            f"logits_prep={profile.get('logits_sample_prep_s', 0) / total:.3f} "
            f"async_enq={profile.get('mx_async_eval_s', 0) / total:.3f} "
            f"sync_wait={profile.get('mx_eval_s', 0) / total:.3f} "
            f"materialize={profile.get('materialize_s', 0) / total:.3f} "
            f"mutation={profile.get('mutation_s', 0) / total:.3f}"
        )
    sys.stderr.write("\n".join(lines) + "\n")


def _record_profile_time(profile: dict[str, Any] | None, key: str, start: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + (time.perf_counter() - start)


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
    async_eval: bool


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
    async_eval: bool = False,
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
        async_eval=async_eval,
    )


def _pending_eval_tensors(pending: _PendingStep) -> tuple[mx.array, ...]:
    arrays: list[mx.array] = [pending.token]
    if pending.token_logprob is not None:
        arrays.append(pending.token_logprob)
    if pending.top_token_ids is not None:
        arrays.append(pending.top_token_ids)
    if pending.top_token_logprobs is not None:
        arrays.append(pending.top_token_logprobs)
    return tuple(arrays)


def _pending_logprob_tensors(pending: _PendingStep) -> tuple[mx.array, ...]:
    arrays: list[mx.array] = []
    if pending.token_logprob is not None:
        arrays.append(pending.token_logprob)
    if pending.top_token_ids is not None:
        arrays.append(pending.top_token_ids)
    if pending.top_token_logprobs is not None:
        arrays.append(pending.top_token_logprobs)
    return tuple(arrays)


def _enqueue_pending_eval(
    pending: _PendingStep,
    *,
    async_eval: bool,
    profile: dict[str, Any] | None,
) -> None:
    arrays = _pending_eval_tensors(pending)
    if async_eval:
        t0 = time.perf_counter()
        cast(Any, mx.async_eval)(*arrays)
        _record_profile_time(profile, "mx_async_eval_s", t0)
        return
    t0 = time.perf_counter()
    mx.eval(*arrays)
    _record_profile_time(profile, "mx_eval_s", t0)


def _sync_arrays(arrays: tuple[mx.array, ...], *, profile: dict[str, Any] | None) -> None:
    if not arrays:
        return
    t0 = time.perf_counter()
    mx.eval(*arrays)
    _record_profile_time(profile, "mx_eval_s", t0)


def _sample_from_logits(
    logits: mx.array,
    *,
    plan: DecodePlan,
    history: _RecentTokenHistory,
    profile: dict[str, Any] | None = None,
    async_eval: bool = False,
) -> _PendingStep:
    if profile is not None:
        t_prep0 = time.perf_counter()
    if plan.logits.enabled:
        logits = plan.logits.apply(history.snapshot(), logits)

    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    token = plan.sampler(logprobs)

    pending = _PendingStep(token=token)

    if plan.emit_logprobs:
        token_logprob = mx.take_along_axis(logprobs, token[:, None], axis=-1)
        pending.token_logprob = token_logprob

        if plan.top_logprobs > 0:
            top_count = min(plan.top_logprobs, logprobs.shape[-1])
            kth = logprobs.shape[-1] - top_count
            top_token_ids = mx.argpartition(logprobs, kth=kth, axis=-1)[:, -top_count:]
            top_token_logprobs = mx.take_along_axis(logprobs, top_token_ids, axis=-1)
            order = mx.argsort(top_token_logprobs, axis=-1)[:, ::-1]
            pending.top_token_ids = mx.take_along_axis(top_token_ids, order, axis=-1)
            pending.top_token_logprobs = mx.take_along_axis(top_token_logprobs, order, axis=-1)

    if profile is not None:
        _record_profile_time(profile, "logits_sample_prep_s", t_prep0)

    _enqueue_pending_eval(pending, async_eval=async_eval, profile=profile)
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


def _materialize_state(
    pending: _PendingStep,
    *,
    plan: DecodePlan,
) -> tuple[int, str, FinishReason | None]:
    token_id = int(pending.token.item())
    text = plan.decoder(token_id) if plan.decode_text else ""

    finish_reason = plan.stop.check_token(token_id)
    if finish_reason is None and plan.stop.needs_text:
        finish_reason = plan.stop.check_text(text)
    return token_id, text, finish_reason


def _build_event(
    pending: _PendingStep,
    *,
    plan: DecodePlan,
    token_id: int,
    text: str,
    finish_reason: FinishReason | None,
    generation_tokens: int,
) -> TokenEvent:
    event = TokenEvent(
        token_id=token_id,
        text=text,
        finish_reason=finish_reason,
        logprobs=_build_logprob_payload(pending, decoder=plan.decoder),
        prompt_tokens=plan.prompt_token_count,
        generation_tokens=generation_tokens,
    )
    return event


def _materialize_event(
    pending: _PendingStep,
    *,
    plan: DecodePlan,
    generation_tokens: int,
) -> tuple[TokenEvent, int]:
    token_id, text, finish_reason = _materialize_state(pending, plan=plan)
    return (
        _build_event(
            pending,
            plan=plan,
            token_id=token_id,
            text=text,
            finish_reason=finish_reason,
            generation_tokens=generation_tokens,
        ),
        token_id,
    )


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
        # Delayed KV quantization replaces the cache object type and internal
        # state layout, which is not a stable boundary for mx.compile.
        forward.downgrade_to_uncompiled(cache)


def decode_loop(
    cache: list[Any],
    first_logits: mx.array,
    *,
    plan: DecodePlan,
    clear_cache_interval: int = 256,
    quantized_kv_start: int = 0,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    profile: dict[str, Any] | None = None,
) -> Iterator[TokenEvent]:
    """Run the staged decode loop, yielding one TokenEvent per generated token."""

    history = _RecentTokenHistory(plan.logits.token_history_size)
    pending = _sample_from_logits(
        first_logits,
        plan=plan,
        history=history,
        profile=profile,
        async_eval=plan.async_eval,
    )

    step_index = 0
    while True:
        generation_tokens = step_index + 1
        current = pending

        if plan.async_eval:
            _sync_arrays((current.token,), profile=profile)
            t_mat0 = time.perf_counter()
            token_id, text, finish_reason = _materialize_state(current, plan=plan)
            history.append(token_id)
            _record_profile_time(profile, "materialize_s", t_mat0)

            if finish_reason is None:
                t_mut0 = time.perf_counter()
                _apply_mutation_boundary(
                    step_index=step_index,
                    cache=cache,
                    forward=plan.forward,
                    clear_cache_interval=clear_cache_interval,
                    quantized_kv_start=quantized_kv_start,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                )
                _record_profile_time(profile, "mutation_s", t_mut0)

                step_index += 1
                lp0 = (
                    float(profile.get("logits_sample_prep_s", 0.0))
                    if profile is not None
                    else 0.0
                )
                async0 = float(profile.get("mx_async_eval_s", 0.0)) if profile is not None else 0.0
                t_fwd0 = time.perf_counter()
                next_logits = plan.forward.forward(mx.reshape(current.token, (1, 1)))
                dt = time.perf_counter() - t_fwd0
                if profile is not None:
                    profile["forward_decode_s"] = float(profile.get("forward_decode_s", 0.0)) + dt
                    profile["n_forward_decode"] = int(profile.get("n_forward_decode", 0)) + 1
                    profile.setdefault("forward_wall_samples", []).append(dt)
                pending = _sample_from_logits(
                    next_logits[:, -1, :],
                    plan=plan,
                    history=history,
                    profile=profile,
                    async_eval=True,
                )
                if profile is not None:
                    d_lp = float(profile.get("logits_sample_prep_s", 0.0)) - lp0
                    d_async = float(profile.get("mx_async_eval_s", 0.0)) - async0
                    profile.setdefault("step_wall_samples", []).append(dt + d_lp + d_async)

            _sync_arrays(_pending_logprob_tensors(current), profile=profile)
            t_evt0 = time.perf_counter()
            event = _build_event(
                current,
                plan=plan,
                token_id=token_id,
                text=text,
                finish_reason=finish_reason,
                generation_tokens=generation_tokens,
            )
            _record_profile_time(profile, "materialize_s", t_evt0)
            yield event

            if finish_reason is not None:
                return
            continue

        t_mat0 = time.perf_counter()
        event, token_id = _materialize_event(
            current,
            plan=plan,
            generation_tokens=generation_tokens,
        )
        history.append(token_id)
        _record_profile_time(profile, "materialize_s", t_mat0)
        yield event

        if event.finish_reason is not None:
            return

        t_mut0 = time.perf_counter()
        _apply_mutation_boundary(
            step_index=step_index,
            cache=cache,
            forward=plan.forward,
            clear_cache_interval=clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )
        _record_profile_time(profile, "mutation_s", t_mut0)

        step_index += 1
        lp0 = float(profile.get("logits_sample_prep_s", 0.0)) if profile is not None else 0.0
        ev0 = float(profile.get("mx_eval_s", 0.0)) if profile is not None else 0.0
        t_fwd0 = time.perf_counter()
        next_logits = plan.forward.forward(mx.reshape(current.token, (1, 1)))
        dt = time.perf_counter() - t_fwd0
        if profile is not None:
            profile["forward_decode_s"] = float(profile.get("forward_decode_s", 0.0)) + dt
            profile["n_forward_decode"] = int(profile.get("n_forward_decode", 0)) + 1
            profile.setdefault("forward_wall_samples", []).append(dt)
        pending = _sample_from_logits(
            next_logits[:, -1, :],
            plan=plan,
            history=history,
            profile=profile,
        )
        if profile is not None:
            d_lp = float(profile.get("logits_sample_prep_s", 0.0)) - lp0
            d_ev = float(profile.get("mx_eval_s", 0.0)) - ev0
            profile.setdefault("step_wall_samples", []).append(dt + d_lp + d_ev)
