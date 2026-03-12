"""Generate module — single-request inference (§6.1, §9, FR3).

Public API: ``generate()`` — takes model, tokenizer, prompt, options
and yields a stream of TokenEvent objects.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import GenerateOptions, TokenEvent
from mlxs.cache.kv import KVCache
from mlxs.generate.decode import decode_loop
from mlxs.generate.logits import make_logits_processors
from mlxs.generate.prefill import chunked_prefill
from mlxs.generate.sampling import make_sampler
from mlxs.generate.stop import StopCondition
from mlxs.protocols.generate import TokenizerProtocol


def generate(
    model: nn.Module,
    tokenizer: TokenizerProtocol,
    prompt: str | list[int],
    options: GenerateOptions | None = None,
    *,
    cache: list[KVCache] | None = None,
    prefill_step_size: int = 2048,
) -> Iterator[TokenEvent]:
    """Generate tokens from a prompt (§6.1, FR3).

    This is the single-request generation entry point. It:
    1. Encodes the prompt (if string).
    2. Creates or reuses KV cache.
    3. Runs chunked prefill.
    4. Runs the decode loop, yielding TokenEvent per token.

    All abstractions (sampler, stop condition, logits processors) are
    resolved once here — not per token (O2).

    Args:
        model: Model satisfying ModelProtocol.
        tokenizer: Tokenizer satisfying TokenizerProtocol.
        prompt: Input text or pre-tokenized token ids.
        options: Generation parameters. Defaults to GenerateOptions().
        cache: Optional pre-populated KV cache (e.g. from prompt cache).
        prefill_step_size: Max tokens per prefill chunk.

    Yields:
        TokenEvent for each generated token. The last event has
        finish_reason set.
    """
    if options is None:
        options = GenerateOptions()

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)

    if not prompt_tokens:
        raise ValueError("Prompt must not be empty")

    prompt_array = mx.array(prompt_tokens)
    prompt_token_count = len(prompt_tokens)

    # Set seed if specified
    if options.seed is not None:
        mx.random.seed(options.seed)

    # Create KV cache if not provided
    if cache is None:
        cache = model.make_cache()

    # Build sampler (resolved once, not per token — O2)
    sampler = make_sampler(
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        min_p=options.min_p,
    )

    # Build logits processors (resolved once)
    logits_processors = make_logits_processors(
        repetition_penalty=options.repetition_penalty,
    )

    # Build stop condition (resolved once)
    stop = StopCondition(
        eos_token_id=tokenizer.eos_token_id,
        max_tokens=options.max_tokens,
        stop_sequences=options.stop_sequences,
        extra_eos_token_ids=options.extra_eos_token_ids,
    )

    # Prefill: process prompt through model
    first_logits = chunked_prefill(
        model,
        prompt_array,
        cache,
        prefill_step_size=prefill_step_size,
    )

    # Decode: generate tokens one at a time
    yield from decode_loop(
        model,
        cache,
        first_logits,
        sampler=sampler,
        stop=stop,
        decoder=tokenizer.decode,
        logits_processors=logits_processors or None,
        options=options,
        prompt_token_count=prompt_token_count,
    )


__all__ = ["generate"]
