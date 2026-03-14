"""Decode loop — token-by-token generation (§6.1, O1, O2).

The hottest path in the library. Design principles:
- Zero per-token Python allocations where possible.
- Abstractions (sampler, stop, logits processors) resolved once before loop.
- Minimal Python between mx.eval calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import GenerateOptions, TokenEvent, TokenLogprobs, TopLogprob
from mlxs.cache.kv import KVCache
from mlxs.generate.logits import LogitsProcessor
from mlxs.generate.stop import StopCondition


def decode_loop(
    model: nn.Module,
    cache: list[KVCache],
    first_logits: mx.array,
    *,
    sampler: Callable[[mx.array], mx.array],
    stop: StopCondition,
    decoder: Callable[[int], str],
    logits_processors: list[LogitsProcessor] | None = None,
    options: GenerateOptions,
    prompt_token_count: int,
    forward_fn: Callable[..., mx.array] | None = None,
    clear_cache_interval: int = 256,
    quantized_kv_start: int = 0,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
) -> Iterator[TokenEvent]:
    """Run the decode loop, yielding TokenEvent per generated token.

    Args:
        model: The model to run.
        cache: KV cache list (one per layer).
        first_logits: Logits from the last prefill token, shape (1, vocab).
        sampler: Sampling function (logprobs → token id).
        stop: Stop condition checker.
        decoder: Token id → text decoder function.
        logits_processors: Optional logits processors.
        options: Generation options (for logprobs config).
        prompt_token_count: Number of prompt tokens (for TokenEvent metadata).
        forward_fn: Optional compiled forward function. Falls back to model()
            if None (AC12 fallback-safe).
        clear_cache_interval: Steps between mx.clear_cache() calls (§6.8).
            0 = disabled. Default: 256.
        quantized_kv_start: Convert cache to quantized after this many decode
            steps. 0 = disabled (FR4).
        kv_bits: Quantization bits (used with quantized_kv_start).
        kv_group_size: Quantization group size (used with quantized_kv_start).

    Yields:
        TokenEvent for each generated token.
    """
    # Resolve forward function once (O2 — no per-token dispatch)
    _forward = forward_fn if forward_fn is not None else model

    # Resolve logprobs config once (O2)
    emit_logprobs = options.logprobs
    n_top_logprobs = options.top_logprobs

    tokens_generated: list[int] = []
    logprobs = first_logits - mx.logsumexp(first_logits, keepdims=True)
    y = sampler(logprobs)
    mx.eval(y, logprobs)

    n = 0
    while True:
        token_id = y.item()
        tokens_generated.append(token_id)
        text = decoder(token_id)

        # Check stop condition
        finish_reason = stop.check(token_id, text)

        # Extract logprobs if requested (FR8)
        token_logprobs_data = None
        if emit_logprobs:
            token_lp = logprobs[token_id].item()
            top_lps: tuple[TopLogprob, ...] = ()
            if n_top_logprobs > 0:
                top_indices = mx.argpartition(logprobs, kth=-n_top_logprobs)[-n_top_logprobs:]
                top_indices = top_indices[mx.argsort(logprobs[top_indices])[::-1]]
                mx.eval(top_indices)
                top_lps = tuple(
                    TopLogprob(
                        token_id=int(idx.item()),
                        token=decoder(int(idx.item())),
                        logprob=float(logprobs[idx].item()),
                    )
                    for idx in top_indices
                )
            token_logprobs_data = TokenLogprobs(
                token_logprob=token_lp,
                top_logprobs=top_lps,
            )

        # Build and yield token event
        event = TokenEvent(
            token_id=token_id,
            text=text,
            finish_reason=finish_reason,
            logprobs=token_logprobs_data,
            prompt_tokens=prompt_token_count,
            generation_tokens=n + 1,
        )
        yield event

        if finish_reason is not None:
            return

        # Periodic cache cleanup (§6.8)
        if clear_cache_interval > 0 and n % clear_cache_interval == 0:
            mx.clear_cache()

        # Convert to quantized cache after quantized_kv_start steps (FR4)
        if quantized_kv_start > 0 and kv_bits is not None and n == quantized_kv_start:
            from mlxs.cache import convert_to_quantized

            cache[:] = convert_to_quantized(
                cache, kv_bits=kv_bits, kv_group_size=kv_group_size
            )

        n += 1

        # Compute next token (§6.1 — mx.eval, not mx.async_eval)
        next_logits = _forward(y[None], cache=cache)
        next_logits = next_logits[:, -1, :]

        # Apply logits processors if any
        if logits_processors:
            all_tokens = (
                mx.array(tokens_generated)
                if tokens_generated
                else mx.array([], dtype=mx.int32)
            )
            for processor in logits_processors:
                next_logits = processor(all_tokens, next_logits)

        logprobs = next_logits - mx.logsumexp(next_logits, keepdims=True)
        y = sampler(logprobs)
        mx.eval(y, logprobs)
