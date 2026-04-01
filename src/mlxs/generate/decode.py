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
from mlxs.generate.capabilities import DecodeCapabilities
from mlxs.generate.compile import DecodeForwardRuntime, make_decode_forward_runtime
from mlxs.generate.logits import LogitsProcessorPlan, make_logits_processor_plan
from mlxs.generate.sampling import SamplerFn, make_sampler
from mlxs.generate.stop import StopCondition

_EMPTY_TOKEN_HISTORY = mx.array([], dtype=mx.int32)
_Profile = dict[str, Any] | None
_SELECTIVE_SPLIT_ASYNC_LONG_PROMPT_TOKENS = 1024
_SELECTIVE_SPLIT_ASYNC_LONG_GENERATION_TOKENS = 256
_SELECTIVE_SPLIT_ASYNC_MIN_TOKEN_WORK = 2048 * 256


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def decode_profile_enabled() -> bool:
    """True when ``MLXS_DECODE_PROFILE`` is set (decode timing to stderr)."""

    return _env_flag("MLXS_DECODE_PROFILE")


def decode_async_eval_enabled() -> bool:
    """True when ``MLXS_DECODE_ASYNC_EVAL`` is set (experimental overlap path)."""

    return _env_flag("MLXS_DECODE_ASYNC_EVAL")


def decode_selective_split_async_enabled() -> bool:
    """True when selective split-async gating is enabled."""

    return _env_flag("MLXS_DECODE_SELECTIVE_SPLIT_ASYNC")


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
        f"token_boundary_mode={profile.get('token_boundary_mode', 'unknown')} "
        f"token_boundary_selection={profile.get('token_boundary_selection', 'unknown')} "
        f"token_boundary_reason={profile.get('token_boundary_reason', 'unknown')} "
        f"forward_steps={n_fwd} "
        f"forward_decode_s={profile.get('forward_decode_s', 0.0):.6f} "
        f"logits_sample_prep_s={profile.get('logits_sample_prep_s', 0.0):.6f} "
        f"sync_enqueue_s={profile.get('sync_enqueue_s', 0.0):.6f} "
        f"sync_wait_token_s={profile.get('sync_wait_token_s', 0.0):.6f} "
        f"sync_wait_event_s={profile.get('sync_wait_event_s', 0.0):.6f} "
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
    lines.append(
        "[MLXS_DECODE_PROFILE] sync payload counts: "
        f"enqueue_calls={int(profile.get('sync_enqueue_calls', 0))} "
        f"enqueue_tensors={int(profile.get('sync_enqueue_tensors', 0))} "
        f"token_wait_calls={int(profile.get('sync_wait_token_calls', 0))} "
        f"token_wait_tensors={int(profile.get('sync_wait_token_tensors', 0))} "
        f"event_wait_calls={int(profile.get('sync_wait_event_calls', 0))} "
        f"event_wait_tensors={int(profile.get('sync_wait_event_tensors', 0))}"
    )
    lines.append(
        "[MLXS_DECODE_PROFILE] token boundary diagnostics: "
        f"steps={int(profile.get('token_boundary_steps', 0))} "
        f"wait_reuses_enqueue={int(profile.get('token_boundary_wait_reuses_enqueue_steps', 0))} "
        f"event_wait_empty={int(profile.get('token_boundary_event_wait_empty_steps', 0))} "
        f"token_only={int(profile.get('token_boundary_token_only_steps', 0))}"
    )
    total = sum(
        float(profile.get(key, 0.0))
        for key in (
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
    sync_total = sum(
        float(profile.get(key, 0.0))
        for key in ("sync_enqueue_s", "sync_wait_token_s", "sync_wait_event_s")
    )
    if sync_total > 0:
        lines.append(
            "[MLXS_DECODE_PROFILE] sync wall split: "
            f"enqueue={profile.get('sync_enqueue_s', 0) / sync_total:.3f} "
            f"token_wait={profile.get('sync_wait_token_s', 0) / sync_total:.3f} "
            f"event_wait={profile.get('sync_wait_event_s', 0) / sync_total:.3f}"
        )
    token_boundary_total = float(profile.get("sync_enqueue_s", 0.0)) + float(
        profile.get("sync_wait_token_s", 0.0)
    )
    if token_boundary_total > 0:
        lines.append(
            "[MLXS_DECODE_PROFILE] token boundary wall split: "
            f"enqueue={profile.get('sync_enqueue_s', 0) / token_boundary_total:.3f} "
            f"token_wait={profile.get('sync_wait_token_s', 0) / token_boundary_total:.3f}"
        )
    sys.stderr.write("\n".join(lines) + "\n")


def _record_profile_time(profile: _Profile, key: str, start: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + (time.perf_counter() - start)


@dataclass(frozen=True, slots=True)
class _PendingSyncPayload:
    """Resolved tensor groups for enqueue and later host-visible waits."""

    enqueue: tuple[mx.array, ...]
    token_wait: tuple[mx.array, ...]
    event_wait: tuple[mx.array, ...]

    def token_wait_reuses_enqueue(self) -> bool:
        if len(self.token_wait) > len(self.enqueue):
            return False
        return all(
            wait is enqueued
            for wait, enqueued in zip(self.token_wait, self.enqueue, strict=False)
        )

    @property
    def has_event_wait(self) -> bool:
        return bool(self.event_wait)

    @property
    def is_token_only_boundary(self) -> bool:
        return (
            len(self.enqueue) == 1
            and self.token_wait_reuses_enqueue()
            and not self.has_event_wait
        )


@dataclass(slots=True)
class _PendingStep:
    token: mx.array
    sync_payload: _PendingSyncPayload
    async_eval: bool = False
    token_logprob: mx.array | None = None
    top_token_ids: mx.array | None = None
    top_token_logprobs: mx.array | None = None


@dataclass(frozen=True, slots=True)
class _TokenBoundaryPolicy:
    """Resolved policy for token-boundary scheduling and host readiness."""

    name: str
    selection: str
    reason: str
    async_eval: bool
    profile_key: str
    _enqueue_pending: Callable[[_PendingStep, _Profile], None]
    _wait_for_host: Callable[[_PendingStep, _Profile], None]

    def enqueue(self, pending: _PendingStep, *, profile: _Profile) -> None:
        if profile is not None:
            profile.setdefault("token_boundary_mode", self.name)
            profile.setdefault("token_boundary_selection", self.selection)
            profile.setdefault("token_boundary_reason", self.reason)
            profile["token_boundary_steps"] = int(profile.get("token_boundary_steps", 0)) + 1
            if pending.sync_payload.token_wait_reuses_enqueue():
                profile["token_boundary_wait_reuses_enqueue_steps"] = int(
                    profile.get("token_boundary_wait_reuses_enqueue_steps", 0)
                ) + 1
            if not pending.sync_payload.has_event_wait:
                profile["token_boundary_event_wait_empty_steps"] = int(
                    profile.get("token_boundary_event_wait_empty_steps", 0)
                ) + 1
            if pending.sync_payload.is_token_only_boundary:
                profile["token_boundary_token_only_steps"] = int(
                    profile.get("token_boundary_token_only_steps", 0)
                ) + 1
        self._enqueue_pending(pending, profile)

    def wait_for_host(self, pending: _PendingStep, *, profile: _Profile) -> None:
        self._wait_for_host(pending, profile)


@dataclass(frozen=True, slots=True)
class _SyncPolicy:
    """Resolved policy for scheduling and synchronizing pending tensors."""

    transition_before_emit: bool
    token_boundary: _TokenBoundaryPolicy
    _sync_for_event: Callable[[_PendingStep, _Profile], None]

    @property
    def async_eval(self) -> bool:
        return self.token_boundary.async_eval

    @property
    def profile_key(self) -> str:
        return self.token_boundary.profile_key

    def enqueue(self, pending: _PendingStep, *, step_index: int, profile: _Profile) -> None:
        del step_index
        pending.async_eval = self.async_eval
        self.token_boundary.enqueue(pending, profile=profile)

    def sync_token_for_host(
        self,
        pending: _PendingStep,
        *,
        step_index: int,
        profile: _Profile,
    ) -> None:
        del step_index
        self.token_boundary.wait_for_host(pending, profile=profile)

    def sync_for_event(
        self,
        pending: _PendingStep,
        *,
        step_index: int,
        profile: _Profile,
    ) -> None:
        del step_index
        self._sync_for_event(pending, profile)

    def transition_before_emit_for_step(self, step_index: int) -> bool:
        del step_index
        return self.transition_before_emit


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


@dataclass(frozen=True, slots=True)
class _TensorStep:
    """Device/tensor-side decode step execution."""

    forward: DecodeForwardRuntime
    logits: LogitsProcessorPlan
    sampler: SamplerFn
    emit_logprobs: bool
    top_logprobs: int

    @property
    def token_history_size(self) -> int:
        return int(self.logits.token_history_size)

    def seed(
        self,
        logits: mx.array,
        *,
        step_index: int,
        history: _RecentTokenHistory,
        sync: _SyncPolicy,
        profile: _Profile = None,
    ) -> _PendingStep:
        pending = _build_pending_from_logits(
            logits,
            tensor_step=self,
            history=history,
            profile=profile,
        )
        sync.enqueue(pending, step_index=step_index, profile=profile)
        return pending

    def advance(
        self,
        token: mx.array,
        *,
        step_index: int,
        history: _RecentTokenHistory,
        sync: _SyncPolicy,
        profile: _Profile = None,
    ) -> _PendingStep:
        sync0 = float(profile.get(sync.profile_key, 0.0)) if profile is not None else 0.0
        lp0 = float(profile.get("logits_sample_prep_s", 0.0)) if profile is not None else 0.0
        t_fwd0 = time.perf_counter()
        next_logits = self.forward.forward(mx.reshape(token, (1, 1)))
        dt = time.perf_counter() - t_fwd0
        if profile is not None:
            profile["forward_decode_s"] = float(profile.get("forward_decode_s", 0.0)) + dt
            profile["n_forward_decode"] = int(profile.get("n_forward_decode", 0)) + 1
            profile.setdefault("forward_wall_samples", []).append(dt)
        pending = self.seed(
            next_logits[:, -1, :],
            step_index=step_index,
            history=history,
            sync=sync,
            profile=profile,
        )
        if profile is not None:
            d_lp = float(profile.get("logits_sample_prep_s", 0.0)) - lp0
            d_sync = float(profile.get(sync.profile_key, 0.0)) - sync0
            profile.setdefault("step_wall_samples", []).append(dt + d_lp + d_sync)
        return pending


@dataclass(frozen=True, slots=True)
class DecodePlan:
    """Resolved single-request decode plan."""

    stop: StopCondition
    decoder: Callable[[int | list[int]], str]
    tensor_step: _TensorStep
    sync: _SyncPolicy
    decode_text: bool
    prompt_token_count: int
    capabilities: DecodeCapabilities


@dataclass(frozen=True, slots=True)
class _SelectiveSplitAsyncConfig:
    min_prompt_tokens: int
    min_generation_tokens: int
    min_token_work: int


@dataclass(frozen=True, slots=True)
class _SelectiveSplitAsyncDecision:
    enabled: bool
    reason: str


def _uses_greedy_sampling(options: GenerateOptions) -> bool:
    return bool(
        options.temperature == 0
        and options.top_p == 1.0
        and options.top_k == 0
        and options.min_p == 0.0
    )


def _selective_split_async_eligible(
    *,
    options: GenerateOptions,
    compile_decode: bool,
    prompt_token_count: int,
    quantized_kv_start: int,
    kv_bits: int | None,
) -> _SelectiveSplitAsyncDecision:
    config = _SelectiveSplitAsyncConfig(
        min_prompt_tokens=_env_int(
            "MLXS_DECODE_SELECTIVE_SPLIT_ASYNC_MIN_PROMPT_TOKENS",
            _SELECTIVE_SPLIT_ASYNC_LONG_PROMPT_TOKENS,
        ),
        min_generation_tokens=_env_int(
            "MLXS_DECODE_SELECTIVE_SPLIT_ASYNC_MIN_GENERATION_TOKENS",
            _SELECTIVE_SPLIT_ASYNC_LONG_GENERATION_TOKENS,
        ),
        min_token_work=_env_int(
            "MLXS_DECODE_SELECTIVE_SPLIT_ASYNC_MIN_TOKEN_WORK",
            _SELECTIVE_SPLIT_ASYNC_MIN_TOKEN_WORK,
        ),
    )
    if not compile_decode or not _uses_greedy_sampling(options):
        return _SelectiveSplitAsyncDecision(False, "compile_or_sampling_excluded")
    if options.logprobs or options.top_logprobs > 0:
        return _SelectiveSplitAsyncDecision(False, "logprob_payload_enabled")
    if options.repetition_penalty != 1.0:
        return _SelectiveSplitAsyncDecision(False, "logits_processors_enabled")
    if options.stop_sequences:
        return _SelectiveSplitAsyncDecision(False, "stop_sequences_enabled")
    if quantized_kv_start > 0 and kv_bits is not None:
        return _SelectiveSplitAsyncDecision(False, "delayed_quantized_kv")
    if prompt_token_count < config.min_prompt_tokens:
        return _SelectiveSplitAsyncDecision(False, "short_prompt")
    if options.max_tokens < config.min_generation_tokens:
        return _SelectiveSplitAsyncDecision(False, "short_generation")
    if prompt_token_count * options.max_tokens < config.min_token_work:
        return _SelectiveSplitAsyncDecision(False, "insufficient_token_work")
    return _SelectiveSplitAsyncDecision(True, "long_regime")


def prepare_decode_plan(
    model: Any,
    cache: list[Any],
    *,
    options: GenerateOptions,
    decoder: Callable[[int | list[int]], str],
    eos_token_id: int | None,
    prompt_token_count: int,
    capabilities: DecodeCapabilities,
    compile_decode: bool = False,
    async_eval: bool = False,
    quantized_kv_start: int = 0,
    kv_bits: int | None = None,
) -> DecodePlan:
    """Resolve the staged decode runtime once before token generation."""

    sync = _make_sync_policy(
        async_eval=async_eval,
        selective_split_async=decode_selective_split_async_enabled(),
        options=options,
        compile_decode=compile_decode,
        prompt_token_count=prompt_token_count,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
    )
    tensor_step = _TensorStep(
        forward=make_decode_forward_runtime(model, cache, compile_decode=compile_decode),
        logits=make_logits_processor_plan(repetition_penalty=options.repetition_penalty),
        sampler=make_sampler(
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            min_p=options.min_p,
        ),
        emit_logprobs=options.logprobs,
        top_logprobs=options.top_logprobs,
    )
    return DecodePlan(
        stop=StopCondition(
            eos_token_id=eos_token_id,
            max_tokens=options.max_tokens,
            stop_sequences=options.stop_sequences,
            extra_eos_token_ids=options.extra_eos_token_ids,
        ),
        decoder=decoder,
        tensor_step=tensor_step,
        sync=sync,
        decode_text=True,
        prompt_token_count=prompt_token_count,
        capabilities=capabilities,
    )

def _make_pending_sync_payload(
    *,
    token: mx.array,
    token_logprob: mx.array | None,
    top_token_ids: mx.array | None,
    top_token_logprobs: mx.array | None,
) -> _PendingSyncPayload:
    enqueue: list[mx.array] = [token]
    event_wait: list[mx.array] = []
    if token_logprob is not None:
        enqueue.append(token_logprob)
        event_wait.append(token_logprob)
    if top_token_ids is not None:
        enqueue.append(top_token_ids)
        event_wait.append(top_token_ids)
    if top_token_logprobs is not None:
        enqueue.append(top_token_logprobs)
        event_wait.append(top_token_logprobs)
    return _PendingSyncPayload(
        enqueue=tuple(enqueue),
        token_wait=(token,),
        event_wait=tuple(event_wait),
    )


def _record_sync_payload(profile: _Profile, scope: str, arrays: tuple[mx.array, ...]) -> None:
    if profile is None or not arrays:
        return
    profile[f"{scope}_calls"] = int(profile.get(f"{scope}_calls", 0)) + 1
    profile[f"{scope}_tensors"] = int(profile.get(f"{scope}_tensors", 0)) + len(arrays)


def _run_sync_boundary(
    arrays: tuple[mx.array, ...],
    *,
    runner: Callable[..., None],
    scope: str,
    profile_time_key: str,
    legacy_time_key: str,
    profile: _Profile,
) -> None:
    if not arrays:
        return
    _record_sync_payload(profile, scope, arrays)
    t0 = time.perf_counter()
    runner(*arrays)
    dt = time.perf_counter() - t0
    if profile is None:
        return
    profile[profile_time_key] = float(profile.get(profile_time_key, 0.0)) + dt
    profile[legacy_time_key] = float(profile.get(legacy_time_key, 0.0)) + dt


def _enqueue_pending_sync(pending: _PendingStep, profile: _Profile) -> None:
    _run_sync_boundary(
        pending.sync_payload.enqueue,
        runner=mx.eval,
        scope="sync_enqueue",
        profile_time_key="sync_enqueue_s",
        legacy_time_key="mx_eval_s",
        profile=profile,
    )


def _enqueue_pending_async(pending: _PendingStep, profile: _Profile) -> None:
    _run_sync_boundary(
        pending.sync_payload.enqueue,
        runner=cast(Any, mx.async_eval),
        scope="sync_enqueue",
        profile_time_key="sync_enqueue_s",
        legacy_time_key="mx_async_eval_s",
        profile=profile,
    )


def _sync_pending_noop(pending: _PendingStep, profile: _Profile) -> None:
    del pending, profile


def _sync_pending_token_for_host(pending: _PendingStep, profile: _Profile) -> None:
    _run_sync_boundary(
        pending.sync_payload.token_wait,
        runner=mx.eval,
        scope="sync_wait_token",
        profile_time_key="sync_wait_token_s",
        legacy_time_key="mx_eval_s",
        profile=profile,
    )


def _sync_pending_for_event(pending: _PendingStep, profile: _Profile) -> None:
    _run_sync_boundary(
        pending.sync_payload.event_wait,
        runner=mx.eval,
        scope="sync_wait_event",
        profile_time_key="sync_wait_event_s",
        legacy_time_key="mx_eval_s",
        profile=profile,
    )


def _make_sync_policy(
    *,
    async_eval: bool,
    selective_split_async: bool,
    options: GenerateOptions,
    compile_decode: bool,
    prompt_token_count: int,
    quantized_kv_start: int,
    kv_bits: int | None,
) -> _SyncPolicy:
    if async_eval and not selective_split_async:
        return _SyncPolicy(
            transition_before_emit=True,
            token_boundary=_TokenBoundaryPolicy(
                name="split_async",
                selection="global_async",
                reason="global_async",
                async_eval=True,
                profile_key="mx_async_eval_s",
                _enqueue_pending=_enqueue_pending_async,
                _wait_for_host=_sync_pending_token_for_host,
            ),
            _sync_for_event=_sync_pending_for_event,
        )
    if async_eval and selective_split_async:
        decision = _selective_split_async_eligible(
            options=options,
            compile_decode=compile_decode,
            prompt_token_count=prompt_token_count,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
        )
        if decision.enabled:
            return _SyncPolicy(
                transition_before_emit=True,
                token_boundary=_TokenBoundaryPolicy(
                    name="split_async",
                    selection="selective_split_async",
                    reason=decision.reason,
                    async_eval=True,
                    profile_key="mx_async_eval_s",
                    _enqueue_pending=_enqueue_pending_async,
                    _wait_for_host=_sync_pending_token_for_host,
                ),
                _sync_for_event=_sync_pending_for_event,
            )
        return _SyncPolicy(
            transition_before_emit=False,
            token_boundary=_TokenBoundaryPolicy(
                name="single_sync",
                selection="selective_fallback_sync",
                reason=decision.reason,
                async_eval=False,
                profile_key="mx_eval_s",
                _enqueue_pending=_enqueue_pending_sync,
                _wait_for_host=_sync_pending_noop,
            ),
            _sync_for_event=_sync_pending_noop,
        )
    return _SyncPolicy(
        transition_before_emit=False,
        token_boundary=_TokenBoundaryPolicy(
            name="single_sync",
            selection="default_sync" if not selective_split_async else "selective_fallback_sync",
            reason="async_disabled",
            async_eval=False,
            profile_key="mx_eval_s",
            _enqueue_pending=_enqueue_pending_sync,
            _wait_for_host=_sync_pending_noop,
        ),
        _sync_for_event=_sync_pending_noop,
    )


def _build_pending_from_logits(
    logits: mx.array,
    *,
    tensor_step: _TensorStep,
    history: _RecentTokenHistory,
    profile: _Profile = None,
) -> _PendingStep:
    if profile is not None:
        t_prep0 = time.perf_counter()

    if tensor_step.logits.enabled:
        logits = tensor_step.logits.apply(history.snapshot(), logits)

    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    token = tensor_step.sampler(logprobs)
    token_logprob: mx.array | None = None
    top_token_ids: mx.array | None = None
    top_token_logprobs: mx.array | None = None

    if tensor_step.emit_logprobs:
        token_logprob = mx.take_along_axis(logprobs, token[:, None], axis=-1)

        if tensor_step.top_logprobs > 0:
            top_count = min(tensor_step.top_logprobs, logprobs.shape[-1])
            kth = logprobs.shape[-1] - top_count
            top_token_ids = mx.argpartition(logprobs, kth=kth, axis=-1)[:, -top_count:]
            top_token_logprobs = mx.take_along_axis(logprobs, top_token_ids, axis=-1)
            order = mx.argsort(top_token_logprobs, axis=-1)[:, ::-1]
            top_token_ids = mx.take_along_axis(top_token_ids, order, axis=-1)
            top_token_logprobs = mx.take_along_axis(
                top_token_logprobs,
                order,
                axis=-1,
            )

    pending = _PendingStep(
        token=token,
        sync_payload=_make_pending_sync_payload(
            token=token,
            token_logprob=token_logprob,
            top_token_ids=top_token_ids,
            top_token_logprobs=top_token_logprobs,
        ),
        token_logprob=token_logprob,
        top_token_ids=top_token_ids,
        top_token_logprobs=top_token_logprobs,
    )

    if profile is not None:
        _record_profile_time(profile, "logits_sample_prep_s", t_prep0)
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
            pending.top_token_ids[0],
            pending.top_token_logprobs[0],
            strict=False,
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


def _materialize_token(
    pending: _PendingStep,
    *,
    plan: DecodePlan,
) -> tuple[int, str]:
    token_id = int(pending.token.item())
    text = plan.decoder(token_id) if plan.decode_text else ""
    return token_id, text


def _resolve_finish_reason(
    *,
    stop: StopCondition,
    token_id: int,
    text: str,
) -> FinishReason | None:
    finish_reason = stop.check_token(token_id)
    if finish_reason is None and stop.needs_text:
        finish_reason = stop.check_text(text)
    return finish_reason


def _materialize_state(
    pending: _PendingStep,
    *,
    plan: DecodePlan,
) -> tuple[int, str, FinishReason | None]:
    token_id, text = _materialize_token(pending, plan=plan)
    finish_reason = _resolve_finish_reason(stop=plan.stop, token_id=token_id, text=text)
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
    return TokenEvent(
        token_id=token_id,
        text=text,
        finish_reason=finish_reason,
        logprobs=_build_logprob_payload(pending, decoder=plan.decoder),
        prompt_tokens=plan.prompt_token_count,
        generation_tokens=generation_tokens,
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


def _transition_decode_step(
    current: _PendingStep,
    *,
    cache: list[Any],
    history: _RecentTokenHistory,
    plan: DecodePlan,
    step_index: int,
    finish_reason: FinishReason | None,
    before_emit: bool,
    clear_cache_interval: int,
    quantized_kv_start: int,
    kv_bits: int | None,
    kv_group_size: int,
    profile: _Profile = None,
) -> tuple[_PendingStep | None, int]:
    if finish_reason is not None or before_emit != plan.sync.transition_before_emit_for_step(
        step_index
    ):
        return None, step_index

    t_mut0 = time.perf_counter()
    _apply_mutation_boundary(
        step_index=step_index,
        cache=cache,
        forward=plan.tensor_step.forward,
        clear_cache_interval=clear_cache_interval,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )
    _record_profile_time(profile, "mutation_s", t_mut0)

    pending = plan.tensor_step.advance(
        current.token,
        step_index=step_index + 1,
        history=history,
        sync=plan.sync,
        profile=profile,
    )
    return pending, step_index + 1


def decode_loop(
    cache: list[Any],
    first_logits: mx.array,
    *,
    plan: DecodePlan,
    clear_cache_interval: int = 256,
    quantized_kv_start: int = 0,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    profile: _Profile = None,
) -> Iterator[TokenEvent]:
    """Run the staged decode loop, yielding one TokenEvent per generated token."""

    history = _RecentTokenHistory(plan.tensor_step.token_history_size)
    pending = plan.tensor_step.seed(
        first_logits,
        step_index=0,
        history=history,
        sync=plan.sync,
        profile=profile,
    )

    step_index = 0
    while True:
        generation_tokens = step_index + 1
        current = pending

        plan.sync.sync_token_for_host(current, step_index=step_index, profile=profile)

        t_mat0 = time.perf_counter()
        token_id, text, finish_reason = _materialize_state(current, plan=plan)
        history.append(token_id)
        _record_profile_time(profile, "materialize_s", t_mat0)

        next_pending, next_step_index = _transition_decode_step(
            current,
            cache=cache,
            history=history,
            plan=plan,
            step_index=step_index,
            finish_reason=finish_reason,
            before_emit=True,
            clear_cache_interval=clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            profile=profile,
        )

        plan.sync.sync_for_event(current, step_index=step_index, profile=profile)

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

        if next_pending is None:
            next_pending, next_step_index = _transition_decode_step(
                current,
                cache=cache,
                history=history,
                plan=plan,
                step_index=step_index,
                finish_reason=finish_reason,
                before_emit=False,
                clear_cache_interval=clear_cache_interval,
                quantized_kv_start=quantized_kv_start,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                profile=profile,
            )
            assert next_pending is not None

        pending = next_pending
        step_index = next_step_index
