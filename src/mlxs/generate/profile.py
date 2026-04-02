"""Low-invasiveness decode diagnostics for benchmark-focused profiling.

Instrumentation is disabled by default. Enable it by setting
``MLXS_DECODE_PROFILE=1`` in the environment before running generation.

The measurements are intentionally modest in their claims:
- host-side wall around ``_forward(...)`` is measured directly
- host-side wall for tensor-tail setup before ``mx.eval`` is measured directly
- ``mx.eval`` wall is measured directly
- the portion of ``mx.eval`` attributable specifically to model compute is not
  directly separable with the current architecture / MLX APIs
"""

from __future__ import annotations

import logging
import os
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_PROFILE_PREFIX = "[MLXS_DECODE_PROFILE]"


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000:.3f}ms"


def _sample_summary(label: str, samples: list[float]) -> str:
    if not samples:
        return f"{label}=n/a"
    return (
        f"{label}: first={_format_ms(samples[0])} "
        f"median={_format_ms(statistics.median(samples))} "
        f"last={_format_ms(samples[-1])} "
        f"sum={sum(samples):.6f}s"
    )


@dataclass(slots=True)
class DecodeProfiler:
    """Collect aggregate decode diagnostics when profiling is enabled."""

    compile_decode_requested: bool
    emit_logprobs: bool
    top_logprobs: int
    boundary_mode: str = "sync"
    compile_build_attempted: bool = False
    compiled_forward_available: bool = False
    compile_build_wall_s: float = 0.0
    compile_fallback_reason: str | None = None
    compile_rebind_attempts: int = 0
    compile_rebind_successes: int = 0
    compile_rebind_failures: int = 0
    compile_rebind_fallback_to_uncompiled: int = 0
    compile_rebind_last_error: str | None = None
    first_compiled_use_forward_step: int | None = None
    first_compiled_use_generation_token: int | None = None
    compiled_forward_calls: int = 0
    uncompiled_forward_calls: int = 0
    logits_processor_steps: int = 0
    token_history_array_rebuild_steps: int = 0
    token_history_array_tokens_total: int = 0
    logprobs_steps: int = 0
    top_logprobs_steps: int = 0
    clear_cache_calls: int = 0
    cache_replacement_events: int = 0
    cache_replacement_while_compiled_active: int = 0
    seed_tensor_tail_s: float = 0.0
    seed_async_enqueue_s: float = 0.0
    seed_sync_eval_s: float = 0.0
    forward_call_s: list[float] = field(default_factory=list)
    post_forward_tensor_s: list[float] = field(default_factory=list)
    async_enqueue_s: list[float] = field(default_factory=list)
    sync_eval_s: list[float] = field(default_factory=list)
    logprob_aux_tensor_s: list[float] = field(default_factory=list)
    logprob_aux_sync_eval_s: list[float] = field(default_factory=list)
    host_materialize_s: list[float] = field(default_factory=list)
    mutation_boundary_s: list[float] = field(default_factory=list)

    def record_compile_build(
        self,
        *,
        duration_s: float,
        available: bool,
        error: str | None,
    ) -> None:
        self.compile_build_attempted = True
        self.compiled_forward_available = available
        self.compile_build_wall_s = duration_s
        self.compile_fallback_reason = error

    def record_forward_call(
        self,
        *,
        duration_s: float,
        using_compiled: bool,
        forward_step: int,
        generation_token: int,
    ) -> None:
        self.forward_call_s.append(duration_s)
        if using_compiled:
            self.compiled_forward_calls += 1
            if self.first_compiled_use_forward_step is None:
                self.first_compiled_use_forward_step = forward_step
                self.first_compiled_use_generation_token = generation_token
        else:
            self.uncompiled_forward_calls += 1

    def record_logits_processors(self, *, token_count: int) -> None:
        self.logits_processor_steps += 1
        self.token_history_array_rebuild_steps += 1
        self.token_history_array_tokens_total += token_count

    def record_cache_clear(self) -> None:
        self.clear_cache_calls += 1

    def record_cache_replacement(self, *, compiled_active: bool) -> None:
        self.cache_replacement_events += 1
        if compiled_active:
            self.cache_replacement_while_compiled_active += 1

    def set_boundary_mode(self, mode: str) -> None:
        self.boundary_mode = mode

    def record_async_enqueue(self, *, duration_s: float, seed: bool) -> None:
        if seed:
            self.seed_async_enqueue_s += duration_s
        else:
            self.async_enqueue_s.append(duration_s)

    def record_compile_rebind(
        self,
        *,
        success: bool,
        error: str | None,
    ) -> None:
        self.compile_rebind_attempts += 1
        if success:
            self.compile_rebind_successes += 1
            self.compile_rebind_last_error = None
        else:
            self.compile_rebind_failures += 1
            self.compile_rebind_fallback_to_uncompiled += 1
            self.compile_rebind_last_error = error

    def log_summary(self) -> None:
        if self.boundary_mode == "async":
            logger.warning(
                "%s NOTE: forward_call_s and post_forward_tensor_s are direct host-side wall "
                "before boundary submission. async_enqueue_s is direct mx.async_eval enqueue "
                "wall. sync_eval_s is explicit wait wall before host reads. The compiled-forward "
                "share inside the wait boundary is not directly separable in the current "
                "architecture.",
                _PROFILE_PREFIX,
            )
        else:
            logger.warning(
                "%s NOTE: forward_call_s and post_forward_tensor_s are direct host-side wall "
                "before sync only. sync_eval_s is direct mx.eval wall and drains all queued "
                "device work since the previous sync, so the compiled-forward share inside that "
                "boundary is not directly separable in the current architecture.",
                _PROFILE_PREFIX,
            )

        logger.warning(
            "%s boundary_mode=%s compile_decode_requested=%s compile_build_attempted=%s "
            "compiled_forward_available=%s compile_build_wall=%s "
            "compile_fallback_to_uncompiled=%s compile_fallback_reason=%s "
            "compile_rebind_attempts=%d compile_rebind_successes=%d "
            "compile_rebind_failures=%d compile_rebind_fallback_to_uncompiled=%d "
            "compile_rebind_last_error=%s "
            "first_compiled_use_forward_step=%s first_compiled_use_generation_token=%s "
            "compiled_forward_calls=%d uncompiled_forward_calls=%d "
            "cache_replacement_events=%d cache_replacement_while_compiled_active=%d",
            _PROFILE_PREFIX,
            self.boundary_mode,
            self.compile_decode_requested,
            self.compile_build_attempted,
            self.compiled_forward_available,
            _format_ms(self.compile_build_wall_s if self.compile_build_attempted else None),
            self.compile_fallback_reason is not None,
            self.compile_fallback_reason or "n/a",
            self.compile_rebind_attempts,
            self.compile_rebind_successes,
            self.compile_rebind_failures,
            self.compile_rebind_fallback_to_uncompiled,
            self.compile_rebind_last_error or "n/a",
            self.first_compiled_use_forward_step
            if self.first_compiled_use_forward_step is not None
            else "n/a",
            self.first_compiled_use_generation_token
            if self.first_compiled_use_generation_token is not None
            else "n/a",
            self.compiled_forward_calls,
            self.uncompiled_forward_calls,
            self.cache_replacement_events,
            self.cache_replacement_while_compiled_active,
        )

        logger.warning(
            "%s aggregate wall: seed_tensor_tail=%.6fs seed_async_enqueue=%.6fs "
            "seed_sync_eval=%.6fs forward_call=%.6fs post_forward_tensor=%.6fs "
            "async_enqueue=%.6fs sync_eval=%.6fs "
            "logprob_aux_tensor=%.6fs logprob_aux_sync_eval=%.6fs "
            "host_materialize=%.6fs mutation_boundary=%.6fs",
            _PROFILE_PREFIX,
            self.seed_tensor_tail_s,
            self.seed_async_enqueue_s,
            self.seed_sync_eval_s,
            sum(self.forward_call_s),
            sum(self.post_forward_tensor_s),
            sum(self.async_enqueue_s),
            sum(self.sync_eval_s),
            sum(self.logprob_aux_tensor_s),
            sum(self.logprob_aux_sync_eval_s),
            sum(self.host_materialize_s),
            sum(self.mutation_boundary_s),
        )

        measured_total = (
            sum(self.forward_call_s)
            + sum(self.post_forward_tensor_s)
            + sum(self.async_enqueue_s)
            + sum(self.sync_eval_s)
            + sum(self.logprob_aux_tensor_s)
            + sum(self.logprob_aux_sync_eval_s)
            + sum(self.host_materialize_s)
            + sum(self.mutation_boundary_s)
        )
        if measured_total > 0:
            logger.warning(
                "%s fraction of measured per-step wall: forward_call=%.3f "
                "post_forward_tensor=%.3f async_enqueue=%.3f sync_eval=%.3f "
                "logprob_aux_tensor=%.3f "
                "logprob_aux_sync_eval=%.3f host_materialize=%.3f mutation_boundary=%.3f",
                _PROFILE_PREFIX,
                sum(self.forward_call_s) / measured_total,
                sum(self.post_forward_tensor_s) / measured_total,
                sum(self.async_enqueue_s) / measured_total,
                sum(self.sync_eval_s) / measured_total,
                sum(self.logprob_aux_tensor_s) / measured_total,
                sum(self.logprob_aux_sync_eval_s) / measured_total,
                sum(self.host_materialize_s) / measured_total,
                sum(self.mutation_boundary_s) / measured_total,
            )

        logger.warning(
            "%s counters: generated_tokens=%d forward_steps=%d logits_processor_steps=%d "
            "token_history_array_rebuild_steps=%d token_history_array_tokens_total=%d "
            "logprobs_steps=%d top_logprobs_steps=%d clear_cache_calls=%d",
            _PROFILE_PREFIX,
            len(self.host_materialize_s),
            len(self.forward_call_s),
            self.logits_processor_steps,
            self.token_history_array_rebuild_steps,
            self.token_history_array_tokens_total,
            self.logprobs_steps,
            self.top_logprobs_steps,
            self.clear_cache_calls,
        )

        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("forward_call", self.forward_call_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("post_forward_tensor", self.post_forward_tensor_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("async_enqueue", self.async_enqueue_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("sync_eval", self.sync_eval_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("host_materialize", self.host_materialize_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("mutation_boundary", self.mutation_boundary_s),
        )


def create_decode_profiler(
    *,
    compile_decode_requested: bool,
    emit_logprobs: bool,
    top_logprobs: int,
) -> DecodeProfiler | None:
    """Create a decode profiler if env-gated diagnostics are enabled."""
    if not _env_truthy("MLXS_DECODE_PROFILE"):
        return None
    return DecodeProfiler(
        compile_decode_requested=compile_decode_requested,
        emit_logprobs=emit_logprobs,
        top_logprobs=top_logprobs,
    )
