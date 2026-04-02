"""Compiled decode and warmup utilities (§6.8, AC12).

Wraps the model forward pass with mx.compile for decode-time optimization.
Provides warmup logic to trigger JIT compilation before serving.

All optimizations are fallback-safe: when disabled, behavior is identical
to the baseline (AC12).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from mlxs.generate.profile import DecodeProfiler
from mlxs.protocols.cache import CacheProtocol

logger = logging.getLogger(__name__)


def make_compiled_step(
    model: nn.Module,
    cache: list[CacheProtocol],
) -> Callable[[mx.array], mx.array]:
    """Create a compiled single-token decode step.

    ``mx.compile`` only accepts array trees as *formal* arguments; ``KVCache``
    objects are not traceable, so the active cache list is **closed over** here
    and the compiled function takes only ``input_ids`` (§6.8, AC12).

    ``decode_loop`` should call the result as ``step(input_ids)`` — same as an
    uncompiled ``functools.partial(model, cache=cache)``.

    Args:
        model: The model to compile.
        cache: KV cache list for this generation (same instance as decode_loop).

    Returns:
        ``(input_ids) -> logits``; only ``input_ids`` participates in compilation.
    """

    @mx.compile
    def compiled_step(input_ids: mx.array) -> mx.array:
        return cast(mx.array, model(input_ids, cache=cache))

    return compiled_step


@dataclass(slots=True)
class ForwardRuntime:
    """Own decode forward dispatch and compile/cache lifecycle."""

    model: nn.Module
    cache: list[CacheProtocol]
    compile_requested: bool
    profiler: DecodeProfiler | None = None
    _compiled_step: Callable[[mx.array], mx.array] | None = None
    _uncompiled_step: Callable[[mx.array], mx.array] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._uncompiled_step = partial(self.model, cache=self.cache)

    @classmethod
    def create(
        cls,
        model: nn.Module,
        cache: list[CacheProtocol],
        *,
        compile_decode: bool,
        profiler: DecodeProfiler | None = None,
    ) -> ForwardRuntime:
        runtime = cls(
            model=model,
            cache=cache,
            compile_requested=compile_decode,
            profiler=profiler,
        )
        if compile_decode:
            runtime._build_initial_compiled_step()
        return runtime

    @property
    def compiled_active(self) -> bool:
        return self._compiled_step is not None

    def forward(
        self,
        input_ids: mx.array,
        *,
        forward_step: int,
        generation_token: int,
    ) -> mx.array:
        t0 = time.perf_counter() if self.profiler is not None else 0.0
        step = self._compiled_step if self._compiled_step is not None else self._uncompiled_step
        logits = step(input_ids)
        if self.profiler is not None:
            self.profiler.record_forward_call(
                duration_s=time.perf_counter() - t0,
                using_compiled=self.compiled_active,
                forward_step=forward_step,
                generation_token=generation_token,
            )
        return logits

    def on_cache_replaced(self, cache: list[CacheProtocol]) -> None:
        self.cache = cache
        self._uncompiled_step = partial(self.model, cache=self.cache)
        if not self.compile_requested or not self.compiled_active:
            return
        try:
            self._compiled_step = make_compiled_step(self.model, self.cache)
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
            self._compiled_step = make_compiled_step(self.model, self.cache)
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
    """Run a dummy forward pass to trigger JIT compilation (§6.8).

    This forces MLX to compile the model graph before actual inference,
    avoiding cold-start latency on the first real request.

    Args:
        model: The model to warm up.
        cache_factory: Callable that returns a fresh cache list (e.g. model.make_cache).
        vocab_size: Vocab size for dummy input.
    """
    logger.info("Running warmup forward pass...")
    cache = cache_factory()

    # Single dummy token — minimal computation
    dummy_input = mx.array([[0]])
    logits = model(dummy_input, cache=cache)
    mx.eval(logits)

    # Clean up warmup state
    del cache, logits, dummy_input
    mx.clear_cache()
    logger.info("Warmup complete")
