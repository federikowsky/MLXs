"""Compile-less eager-first greedy runtime slice."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import GenerateOptions
from mlxs.cache.kv import KVCache
from mlxs.generate.core import require_async_eval

EagerGreedyStepFn = Callable[[mx.array], mx.array]


def supports_eager_greedy_slice(
    *,
    compile_decode: bool,
    options: GenerateOptions,
    cache: list[KVCache],
    quantized_kv_start: int,
    kv_bits: int | None,
) -> bool:
    """Return True when the narrow eager-first greedy contract is satisfied."""
    return (
        not compile_decode
        and options.max_tokens > 0
        and options.temperature == 0.0
        and options.top_p == 1.0
        and options.top_k == 0
        and options.min_p == 0.0
        and options.repetition_penalty == 1.0
        and not options.logprobs
        and options.top_logprobs == 0
        and quantized_kv_start == 0
        and kv_bits is None
        and all(type(layer_cache) is KVCache for layer_cache in cache)
    )


def build_eager_greedy_step_fn(
    model: nn.Module,
    cache: list[KVCache],
    stream: mx.Stream,
) -> EagerGreedyStepFn:
    """Build the eager-first greedy step closure for plain KV cache decode."""

    def step_fn(prev_token: mx.array) -> mx.array:
        with mx.stream(stream):
            logits = cast(mx.array, model(prev_token[None], cache=cache))
            logits = logits[:, -1, :]
            return mx.argmax(logits, axis=-1).astype(mx.int32).reshape(-1)

    return step_fn


@dataclass(slots=True)
class EagerGreedyRuntime:
    """Minimal runtime that owns a greedy-only eager decode loop."""

    cache: list[KVCache]
    stream: mx.Stream
    clear_cache_interval: int
    step_fn: EagerGreedyStepFn
    async_eval: Callable[..., object]

    def decode(
        self,
        first_logits: mx.array,
        max_tokens: int,
    ) -> Generator[tuple[int, None], None, None]:
        with mx.stream(self.stream):
            seed_token = mx.argmax(first_logits, axis=-1).astype(mx.int32).reshape(-1)

        self.async_eval(seed_token)
        mx.eval(seed_token)
        yield int(seed_token.item()), None
        self._apply_mutation_boundary(yielded_count=1)

        if max_tokens <= 1:
            return

        current_token = self.step_fn(seed_token)
        self.async_eval(current_token)
        yielded_count = 1

        while True:
            if yielded_count + 1 == max_tokens:
                yield int(current_token.item()), None
                return

            next_token = self.step_fn(current_token)
            self.async_eval(next_token)
            yield int(current_token.item()), None

            yielded_count += 1
            self._apply_mutation_boundary(yielded_count=yielded_count)
            current_token = next_token

    def _apply_mutation_boundary(self, *, yielded_count: int) -> None:
        if self.clear_cache_interval > 0 and (yielded_count - 1) % self.clear_cache_interval == 0:
            mx.clear_cache()


def build_eager_greedy_runtime(
    model: nn.Module,
    cache: list[KVCache],
    stream: mx.Stream,
    clear_cache_interval: int,
) -> EagerGreedyRuntime:
    """Build the narrow eager-first greedy runtime for Gate 1."""
    if any(type(layer_cache) is not KVCache for layer_cache in cache):
        raise NotImplementedError("Eager-first greedy slice requires plain KVCache layers")

    return EagerGreedyRuntime(
        cache=cache,
        stream=stream,
        clear_cache_interval=clear_cache_interval,
        step_fn=build_eager_greedy_step_fn(model, cache, stream),
        async_eval=require_async_eval(),
    )


__all__ = [
    "EagerGreedyRuntime",
    "EagerGreedyStepFn",
    "build_eager_greedy_runtime",
    "build_eager_greedy_step_fn",
    "supports_eager_greedy_slice",
]
