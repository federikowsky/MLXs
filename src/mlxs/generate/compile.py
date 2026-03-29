"""Compiled decode and warmup utilities (§6.8, AC12).

Wraps the model forward pass with mx.compile for decode-time optimization.
Provides warmup logic to trigger JIT compilation before serving.

All optimizations are fallback-safe: when disabled, behavior is identical
to the baseline (AC12).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import mlx.core as mx

logger = logging.getLogger(__name__)

DecodeStep = Callable[[mx.array], mx.array]


def make_compiled_step(
    model: Any,
    cache: list[Any],
) -> DecodeStep:
    """Create a compiled single-token decode step.

    ``mx.compile`` only accepts array trees as *formal* arguments; ``KVCache``
    objects are not traceable, so the active cache list is **closed over** here
    and the compiled function takes only ``input_ids`` (§6.8, AC12). That makes
    the compiled step request-local: it is tied to one active cache object graph.

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


class DecodeForwardRuntime:
    """Decode-time forward runtime with explicit cache rebinding."""

    __slots__ = ("_cache", "_compile_decode", "_compiled", "_model", "_step")

    def __init__(
        self,
        model: Any,
        cache: list[Any],
        *,
        compile_decode: bool,
    ) -> None:
        self._model = model
        self._cache = cache
        self._compile_decode = compile_decode
        self._compiled = False
        self._step: DecodeStep = self._make_uncompiled_step(cache)

    @property
    def compiled(self) -> bool:
        return self._compiled

    def forward(self, input_ids: mx.array) -> mx.array:
        if self._compile_decode and not self._compiled:
            self._try_enable_compiled()

        if not self._compiled:
            return self._step(input_ids)

        try:
            return self._step(input_ids)
        except Exception:
            logger.warning("Compiled decode step failed; falling back to uncompiled forward.")
            self._compile_decode = False
            self._reset_uncompiled_step(self._cache)
            return self._step(input_ids)

    def on_cache_replaced(self, cache: list[Any]) -> None:
        self._cache = cache
        self._reset_uncompiled_step(cache)

    def downgrade_to_uncompiled(self, cache: list[Any]) -> None:
        """Explicitly drop to uncompiled forward for the current request."""

        self._cache = cache
        self._compile_decode = False
        self._reset_uncompiled_step(cache)

    def _try_enable_compiled(self) -> None:
        try:
            self._step = make_compiled_step(self._model, self._cache)
            self._compiled = True
        except Exception:
            logger.warning("Compiled decode setup failed; using uncompiled forward.")
            self._compile_decode = False
            self._reset_uncompiled_step(self._cache)

    def _reset_uncompiled_step(self, cache: list[Any]) -> None:
        self._step = self._make_uncompiled_step(cache)
        self._compiled = False

    def _make_uncompiled_step(self, cache: list[Any]) -> DecodeStep:
        return partial(self._model, cache=cache)


def make_decode_forward_runtime(
    model: Any,
    cache: list[Any],
    *,
    compile_decode: bool,
) -> DecodeForwardRuntime:
    """Create the decode forward runtime for a single request."""

    return DecodeForwardRuntime(model, cache, compile_decode=compile_decode)


def warmup(
    model: Any,
    cache_factory: Callable[[], list[Any]],
    *,
    compile_decode: bool = False,
    vocab_size: int = 32000,
) -> None:
    """Run a dummy forward pass to trigger JIT compilation (§6.8).

    When ``compile_decode`` is enabled, this exercises the same request-local
    decode runtime seam used by ``generate()``. It does not create a reusable
    cross-request compiled step; those steps close over request-local caches.

    Args:
        model: The model to warm up.
        cache_factory: Callable that returns a fresh cache list (e.g. model.make_cache).
        compile_decode: Whether to warm the request-local compiled decode seam.
        vocab_size: Vocab size for dummy input.
    """
    logger.info("Running warmup forward pass...")
    cache = cache_factory()

    # Single dummy token — minimal computation
    dummy_input = mx.array([[0]])
    if compile_decode:
        runtime = make_decode_forward_runtime(model, cache, compile_decode=True)
        logits = runtime.forward(dummy_input)
    else:
        logits = model(dummy_input, cache=cache)
    mx.eval(logits)

    # Clean up warmup state
    del cache, logits, dummy_input
    mx.clear_cache()
    logger.info("Warmup complete")
