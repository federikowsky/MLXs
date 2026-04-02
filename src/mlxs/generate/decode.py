"""Thin wrapper for the decode engine."""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx

from mlxs._types import TokenEvent
from mlxs.generate.compile import StepBackend
from mlxs.generate.profile import DecodeProfiler
from mlxs.generate.runtime import EnginePlan, run_decode_engine
from mlxs.protocols.cache import CacheProtocol


def decode_loop(
    prompt_tokens: mx.array,
    *,
    plan: EnginePlan,
    cache: list[CacheProtocol],
    step_backend: StepBackend,
    input_embeddings: mx.array | None = None,
    prefill_step_size: int = 2048,
    profiler: DecodeProfiler | None = None,
) -> Iterator[TokenEvent]:
    """Run the decode engine for one request."""
    return run_decode_engine(
        prompt_tokens,
        plan=plan,
        cache=cache,
        step_backend=step_backend,
        input_embeddings=input_embeddings,
        prefill_step_size=prefill_step_size,
        profiler=profiler,
    )

