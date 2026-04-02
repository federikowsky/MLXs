"""Single-request generation entrypoint."""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn

from mlxs._errors import InvalidPromptError
from mlxs._types import GenerateOptions, TokenEvent
from mlxs.cache.kv import KVCache
from mlxs.generate.compile import StepBackend
from mlxs.generate.decode import decode_loop
from mlxs.generate.logits import create_step_recipe
from mlxs.generate.profile import create_decode_profiler
from mlxs.generate.runtime import EnginePlan
from mlxs.generate.stop import StopMatcher
from mlxs.protocols.generate import TokenizerProtocol


def generate(
    model: nn.Module,
    tokenizer: TokenizerProtocol,
    prompt: str | list[int],
    options: GenerateOptions | None = None,
    *,
    cache: list[KVCache] | None = None,
    input_embeddings: mx.array | None = None,
    prefill_step_size: int = 2048,
    compile_decode: bool = False,
    clear_cache_interval: int = 256,
    quantized_kv_start: int = 0,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    final_cache_out: list[list[KVCache]] | None = None,
) -> Iterator[TokenEvent]:
    """Generate tokens from a prompt."""
    if options is None:
        options = GenerateOptions()

    prompt_tokens = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    if not prompt_tokens:
        raise ValueError("Prompt must not be empty")

    if input_embeddings is not None:
        if input_embeddings.ndim != 2:
            raise InvalidPromptError(
                f"input_embeddings must be 2-D (T, D), got shape {input_embeddings.shape}"
            )
        if input_embeddings.shape[0] != len(prompt_tokens):
            raise InvalidPromptError(
                f"input_embeddings length ({input_embeddings.shape[0]}) must match "
                f"prompt length ({len(prompt_tokens)})"
            )

    prompt_array = mx.array(prompt_tokens)
    prompt_token_count = len(prompt_tokens)

    if options.seed is not None:
        mx.random.seed(options.seed)

    if cache is None:
        cache = model.make_cache()

    recipe = create_step_recipe(
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        min_p=options.min_p,
        repetition_penalty=options.repetition_penalty,
        emit_logprobs=options.logprobs,
        top_logprobs=options.top_logprobs,
    )
    stop = StopMatcher(
        eos_token_id=tokenizer.eos_token_id,
        max_tokens=options.max_tokens,
        stop_sequences=options.stop_sequences,
        extra_eos_token_ids=options.extra_eos_token_ids,
    )
    profiler = create_decode_profiler(
        compile_decode_requested=compile_decode,
        emit_logprobs=options.logprobs,
        top_logprobs=options.top_logprobs,
    )
    plan = EnginePlan(
        recipe=recipe,
        stop=stop,
        decoder=tokenizer.decode,
        prompt_token_count=prompt_token_count,
        clear_cache_interval=clear_cache_interval,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )
    step_backend = StepBackend.create(
        model,
        cache,
        recipe=recipe,
        compile_decode=compile_decode,
        profiler=profiler,
    )

    def _gen() -> Iterator[TokenEvent]:
        try:
            yield from decode_loop(
                prompt_array,
                plan=plan,
                cache=cache,
                step_backend=step_backend,
                input_embeddings=input_embeddings,
                prefill_step_size=prefill_step_size,
                profiler=profiler,
            )
        finally:
            if final_cache_out is not None:
                final_cache_out.append(cache)

    return _gen()


__all__ = ["generate"]

