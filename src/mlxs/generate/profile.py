"""Decode Engine V4 wrapper-based profiling helpers."""

from __future__ import annotations

import logging
import os
import statistics
from collections.abc import Generator
from dataclasses import dataclass, field
from time import perf_counter

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


def _quantile(samples: list[float], q: float) -> float | None:
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    return statistics.quantiles(samples, n=100, method="inclusive")[int(q * 100) - 1]


def _sample_summary(label: str, samples: list[float]) -> str:
    if not samples:
        return f"{label}=n/a"
    return (
        f"{label}: count={len(samples)} "
        f"p05={_format_ms(_quantile(samples, 0.05))} "
        f"p50={_format_ms(statistics.median(samples))} "
        f"p95={_format_ms(_quantile(samples, 0.95))} "
        f"sum={sum(samples):.6f}s"
    )


@dataclass(slots=True)
class DecodeProfiler:
    """V4 profiling state collected outside the production core loop."""

    compile_decode_requested: bool
    emit_logprobs: bool
    top_logprobs: int
    detail_mode: bool
    prefill_wall_s: float | None = None
    first_token_wall_s: float | None = None
    per_step_wall_s: list[float] = field(default_factory=list)
    step_fn_wall_s: list[float] = field(default_factory=list)
    async_eval_wall_s: list[float] = field(default_factory=list)
    item_wait_wall_s: list[float] = field(default_factory=list)
    tokenizer_wall_s: list[float] = field(default_factory=list)
    stop_check_wall_s: list[float] = field(default_factory=list)
    event_build_wall_s: list[float] = field(default_factory=list)

    def log_summary(self) -> None:
        logger.warning(
            "%s boundary proxy note: item_wait_wall_s conflates device compute, "
            "synchronization, and scalar host read latency.",
            _PROFILE_PREFIX,
        )
        logger.warning(
            "%s detail_mode=%s compile_decode_requested=%s emit_logprobs=%s "
            "top_logprobs=%d prefill_wall_s=%s first_token_wall_s=%s",
            _PROFILE_PREFIX,
            self.detail_mode,
            self.compile_decode_requested,
            self.emit_logprobs,
            self.top_logprobs,
            _format_ms(self.prefill_wall_s),
            _format_ms(self.first_token_wall_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("per_step_wall_s", self.per_step_wall_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("step_fn_wall_s", self.step_fn_wall_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("async_eval_wall_s", self.async_eval_wall_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("item_wait_wall_s", self.item_wait_wall_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("tokenizer_wall_s", self.tokenizer_wall_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("stop_check_wall_s", self.stop_check_wall_s),
        )
        logger.warning(
            "%s %s",
            _PROFILE_PREFIX,
            _sample_summary("event_build_wall_s", self.event_build_wall_s),
        )


def create_decode_profiler(
    *,
    compile_decode_requested: bool,
    emit_logprobs: bool,
    top_logprobs: int,
) -> DecodeProfiler | None:
    """Create a V4 decode profiler if env-gated diagnostics are enabled."""
    if not _env_truthy("MLXS_DECODE_PROFILE"):
        return None
    return DecodeProfiler(
        compile_decode_requested=compile_decode_requested,
        emit_logprobs=emit_logprobs,
        top_logprobs=top_logprobs,
        detail_mode=_env_truthy("MLXS_DECODE_PROFILE_DETAIL"),
    )


def profile_steps(
    core_iter: Generator[tuple[int, object], None, None],
    profiler: DecodeProfiler,
) -> Generator[tuple[int, object], None, None]:
    """Summary-mode wrapper around the core iterator."""
    first_token = True
    try:
        while True:
            t0 = perf_counter()
            yielded = next(core_iter)
            dt = perf_counter() - t0
            profiler.per_step_wall_s.append(dt)
            if first_token:
                profiler.first_token_wall_s = dt
                first_token = False
            yield yielded
    except StopIteration:
        return
    finally:
        profiler.log_summary()


__all__ = ["DecodeProfiler", "create_decode_profiler", "profile_steps"]
