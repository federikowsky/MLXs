"""Generate module — Decode Engine V4 public entrypoint."""

from __future__ import annotations

import time
from collections.abc import Generator, Iterator

import mlx.core as mx
import mlx.nn as nn

import mlxs.generate.eager_greedy as eager_greedy_mod
from mlxs._errors import InvalidPromptError
from mlxs._types import GenerateOptions, TokenEvent
from mlxs.cache.kv import KVCache
from mlxs.generate.compile import warmup
from mlxs.generate.core import (
    _decode_steps,
    _decode_steps_detail,
    _decode_steps_explicit_state,
    _generation_stream,
    require_async_eval,
)
from mlxs.generate.events import token_event_stream
from mlxs.generate.logits import make_logits_processors
from mlxs.generate.prefill import chunked_prefill
from mlxs.generate.profile import create_decode_profiler, profile_steps
from mlxs.generate.recipe import CompileMode, Recipe
from mlxs.generate.runtime import DecodePlan
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

    _ensure_m4_group4_supported(
        options=options,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
    )
    require_async_eval()

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

    if options.seed is not None:
        mx.random.seed(options.seed)

    if cache is None:
        cache = model.make_cache()

    if (
        quantized_kv_start > 0
        and kv_bits is not None
        and any(type(layer_cache) is not KVCache for layer_cache in cache)
    ):
        raise NotImplementedError(
            "Decode Engine V5 M4 Group 4 delayed quantized KV requires plain KVCache layers"
        )

    if compile_decode:
        _ensure_m3b_compile_supported(
            model,
            cache,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
        )

    if compile_decode:
        warmup(model, model.make_cache)

    stop = StopCondition(
        eos_token_id=tokenizer.eos_token_id,
        max_tokens=options.max_tokens,
        stop_sequences=options.stop_sequences,
        extra_eos_token_ids=options.extra_eos_token_ids,
    )
    plan = DecodePlan(
        stop=stop,
        decoder=tokenizer.decode,
        prompt_token_count=len(prompt_tokens),
        clear_cache_interval=clear_cache_interval,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )
    logits_processors = tuple(
        make_logits_processors(
            repetition_penalty=options.repetition_penalty,
        )
    )
    recipe = Recipe(
        sampler=make_sampler(
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            min_p=options.min_p,
        ),
        logits_processors=logits_processors,
        has_processors=bool(logits_processors),
        processor_context_size=20 if logits_processors else 0,
        emit_logprobs=options.logprobs or options.top_logprobs > 0,
        emit_top_logprobs=options.top_logprobs > 0,
        top_logprobs_k=options.top_logprobs,
        compile_mode=CompileMode.ON if compile_decode else CompileMode.OFF,
    )
    profiler = create_decode_profiler(
        compile_decode_requested=compile_decode,
        emit_logprobs=options.logprobs,
        top_logprobs=options.top_logprobs,
    )
    eager_greedy_slice_active = eager_greedy_mod.supports_eager_greedy_slice(
        compile_decode=compile_decode,
        options=options,
        cache=cache,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
    ) and not (profiler is not None and profiler.detail_mode)

    prompt_array = mx.array(prompt_tokens)
    prefill_t0 = time.perf_counter()
    first_logits = chunked_prefill(
        model,
        prompt_array,
        cache,
        prefill_step_size=prefill_step_size,
        input_embeddings=input_embeddings,
        stream=_generation_stream,
    )
    if profiler is not None:
        profiler.prefill_wall_s = time.perf_counter() - prefill_t0

    core_iter: Generator[tuple[int, object], None, None]
    compiled_final_cache_holder: list[list[KVCache] | None] | None = None
    if compile_decode:
        compiled_final_cache_holder = [None]
        core_iter = _decode_steps_explicit_state(
            model,
            cache,
            recipe,
            _generation_stream,
            first_logits,
            options.max_tokens,
            clear_cache_interval,
            final_cache_holder=compiled_final_cache_holder,
        )
    elif eager_greedy_slice_active:
        core_iter = eager_greedy_mod.build_eager_greedy_runtime(
            model,
            cache,
            _generation_stream,
            clear_cache_interval,
        ).decode(first_logits, options.max_tokens)
    elif profiler is not None and profiler.detail_mode:
        core_iter = _decode_steps_detail(
            model,
            cache,
            recipe,
            _generation_stream,
            first_logits,
            options.max_tokens,
            clear_cache_interval,
            profiler,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )
    else:
        core_iter = _decode_steps(
            model,
            cache,
            recipe,
            _generation_stream,
            first_logits,
            options.max_tokens,
            clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )

    if profiler is not None:
        core_iter = profile_steps(core_iter, profiler)

    def _gen() -> Iterator[TokenEvent]:
        try:
            yield from token_event_stream(
                core_iter,
                plan,
                recipe,
                options.max_tokens,
                profiler=profiler,
            )
        finally:
            if final_cache_out is not None:
                if compile_decode and compiled_final_cache_holder is not None:
                    final_cache_out.append(compiled_final_cache_holder[0] or cache)
                else:
                    final_cache_out.append(cache)

    return _gen()


def _ensure_m4_group4_supported(
    *,
    options: GenerateOptions,
    quantized_kv_start: int,
    kv_bits: int | None,
) -> None:
    del options
    if kv_bits is not None and quantized_kv_start <= 0:
        raise NotImplementedError(
            "Decode Engine V5 M4 Group 4 supports only delayed quantized KV "
            "(quantized_kv_start > 0)"
        )


def _ensure_m3b_compile_supported(
    model: nn.Module,
    cache: list[KVCache],
    *,
    quantized_kv_start: int,
    kv_bits: int | None,
) -> None:
    if any(type(layer_cache) is not KVCache for layer_cache in cache):
        raise NotImplementedError(
            "Decode Engine V5 M3b compile-on supports only plain KVCache layers"
        )

    if quantized_kv_start > 0 and kv_bits is not None:
        raise NotImplementedError(
            "Decode Engine V5 M4 Group 4 compile-on does not support delayed quantized KV: "
            "the P2 explicit-state decode path carries full-precision array state only and "
            "cannot replace mid-loop with QuantizedKVCache without a new explicit "
            "quantized-state protocol"
        )

    sliding_window = getattr(getattr(model, "model", None), "sliding_window", None)
    if sliding_window is not None:
        raise NotImplementedError(
            "Decode Engine V5 M3b compile-on does not support sliding-window cache paths"
        )


__all__ = ["generate"]
