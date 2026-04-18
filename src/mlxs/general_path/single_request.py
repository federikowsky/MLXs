"""Layer 2 single-request rich generation semantics above Layer 1."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._errors import InvalidPromptError
from mlxs._types import FinishReason, GenerateOptions, TokenEvent, TokenLogprobs, TopLogprob
from mlxs.cache.kv import KVCache
from mlxs.general_path.finish import map_core_finish_reason
from mlxs.general_path.stop import StopSequenceMatcher
from mlxs.generate.compile import make_compiled_step
from mlxs.generate.logits import make_logits_processors
from mlxs.generate.sampling import make_sampler
from mlxs.protocols.generate import TokenizerProtocol
from mlxs.runtime_core import CoreExecutionPolicy, CoreState, CoreTerminationPolicy
from mlxs.runtime_core.decode import (
    decode_step,
    materialize_prepared_step,
    prepare_decode_step,
    prepare_next_logits,
    schedule_next_decode_step,
)
from mlxs.runtime_core.prefill import run_prefill

SHORT_LOGPROBS_PREPARED_STEP_THRESHOLD = 512


def _normalize_prompt_tokens(
    tokenizer: TokenizerProtocol,
    prompt: str | list[int],
) -> list[int]:
    return tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)


def _validate_embeddings(prompt_tokens: list[int], input_embeddings: mx.array | None) -> None:
    if input_embeddings is None:
        return
    if input_embeddings.ndim != 2:
        raise InvalidPromptError(
            f"input_embeddings must be 2-D (T, D), got shape {input_embeddings.shape}"
        )
    if input_embeddings.shape[0] != len(prompt_tokens):
        raise InvalidPromptError(
            f"input_embeddings length ({input_embeddings.shape[0]}) must match "
            f"prompt length ({len(prompt_tokens)})"
        )


def _build_top_logprobs(
    tokenizer: TokenizerProtocol,
    logprobs_row: mx.array,
    *,
    top_n: int,
) -> tuple[TopLogprob, ...]:
    if top_n <= 0:
        return ()

    top_indices = mx.argsort(logprobs_row)[::-1][:top_n]
    top_logprobs = logprobs_row[top_indices]
    mx.eval(top_indices, top_logprobs)

    token_ids = [int(token_id) for token_id in top_indices.tolist()]
    logprob_values = [float(logprob) for logprob in top_logprobs.tolist()]

    inner = getattr(tokenizer, "inner", None)
    if inner is not None and hasattr(inner, "batch_decode"):
        tokens = inner.batch_decode([[token_id] for token_id in token_ids])
    else:
        tokens = [tokenizer.decode(token_id) for token_id in token_ids]

    return tuple(
        TopLogprob(token_id=token_id, token=token, logprob=logprob)
        for token_id, token, logprob in zip(token_ids, tokens, logprob_values, strict=True)
    )


def _build_token_logprobs(
    tokenizer: TokenizerProtocol,
    logprobs: mx.array,
    token_id: int,
    *,
    top_n: int,
) -> TokenLogprobs:
    row = logprobs[0]
    token_logprob = float(row[token_id].item())
    return TokenLogprobs(
        token_logprob=token_logprob,
        top_logprobs=_build_top_logprobs(tokenizer, row, top_n=top_n),
    )


def _select_token_from_logits(
    sampler: Any,
    logits: mx.array,
) -> mx.array:
    """Layer 2 sampling consumes normalized log-probabilities, not raw logits."""
    return sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))


def _use_prepared_step_enriched_path(
    *,
    prompt_token_count: int,
    options: GenerateOptions,
    logits_processors: list[Any],
) -> bool:
    if logits_processors or not options.logprobs:
        return False
    if options.top_logprobs > 0:
        return True
    return prompt_token_count <= SHORT_LOGPROBS_PREPARED_STEP_THRESHOLD


def _use_processor_prepared_step_path(
    *,
    prompt_token_count: int,
    options: GenerateOptions,
    logits_processors: list[Any],
    execution_stream: Any | None,
) -> bool:
    return (
        execution_stream is not None
        and bool(logits_processors)
        and not options.logprobs
        and prompt_token_count <= SHORT_LOGPROBS_PREPARED_STEP_THRESHOLD
    )


def generate_single_request(
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
    execution_stream: Any | None = None,
    final_cache_out: list[list[KVCache]] | None = None,
) -> Iterator[TokenEvent]:
    """Run the canonical Layer 2 single-request generation path.

    Layer 2 owns semantic enrichment only: richer sampling, logits processors,
    stop-sequence handling, finish-reason mapping, and optional logprob shaping.
    The underlying prompt ingestion, decode progression, cache ownership, and
    execution policy remain owned by Layer 1.
    """
    if options is None:
        options = GenerateOptions()

    prompt_tokens = _normalize_prompt_tokens(tokenizer, prompt)
    if not prompt_tokens:
        raise ValueError("Prompt must not be empty")
    _validate_embeddings(prompt_tokens, input_embeddings)

    prompt_array = mx.array(prompt_tokens)
    prompt_token_count = len(prompt_tokens)

    if options.seed is not None:
        mx.random.seed(options.seed)

    state = CoreState.adopt(cache) if cache is not None else CoreState.create(model)
    execution = CoreExecutionPolicy(clear_cache_interval=clear_cache_interval)
    eos_token_ids = tuple(
        token_id
        for token_id in (
            tokenizer.eos_token_id,
            *options.extra_eos_token_ids,
        )
        if token_id is not None
    )
    termination = CoreTerminationPolicy(
        max_tokens=options.max_tokens,
        eos_token_ids=eos_token_ids,
    )

    sampler = make_sampler(
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        min_p=options.min_p,
    )
    logits_processors = make_logits_processors(
        repetition_penalty=options.repetition_penalty,
    )
    stop_sequences = StopSequenceMatcher(options.stop_sequences)
    step_fn = make_compiled_step(model, state.cache) if compile_decode else None
    use_prepared_step_enriched_path = _use_prepared_step_enriched_path(
        prompt_token_count=prompt_token_count,
        options=options,
        logits_processors=logits_processors,
    )
    use_processor_prepared_step_path = _use_processor_prepared_step_path(
        prompt_token_count=prompt_token_count,
        options=options,
        logits_processors=logits_processors,
        execution_stream=execution_stream,
    )
    execution = CoreExecutionPolicy(
        clear_cache_interval=clear_cache_interval,
        stream=(
            execution_stream
            if use_prepared_step_enriched_path or use_processor_prepared_step_path
            else None
        ),
    )

    generated_tokens: list[int] = []

    try:
        try:
            logits = run_prefill(
                model,
                prompt_array,
                state,
                execution=execution,
                prefill_step_size=prefill_step_size,
                input_embeddings=input_embeddings,
            )
        except ValueError as exc:
            raise InvalidPromptError(str(exc)) from exc

        if use_prepared_step_enriched_path:
            prepared = prepare_decode_step(
                logits,
                execution=execution,
                select_token=lambda step_logits: _select_token_from_logits(sampler, step_logits),
            )

            while True:
                step_logits = prepared.logits
                logprobs = step_logits - mx.logsumexp(step_logits, axis=-1, keepdims=True)
                next_prepared = None
                if len(generated_tokens) + 1 < options.max_tokens:
                    next_prepared = schedule_next_decode_step(
                        model,
                        state,
                        prepared,
                        execution=execution,
                        select_token=lambda step_logits: _select_token_from_logits(
                            sampler,
                            step_logits,
                        ),
                        step_fn=step_fn,
                    )

                result = materialize_prepared_step(
                    state,
                    prepared,
                    termination=termination,
                    execution=execution,
                    force_eval=not generated_tokens,
                )
                next_logits = None if next_prepared is None else next_prepared.logits
                prepared = (
                    next_prepared
                    if next_prepared is not None
                    else SimpleNamespace(logits=mx.array([]), token=mx.array([]))
                )

                token_id = result.token_id
                generated_tokens.append(token_id)
                text = tokenizer.decode(token_id)
                finish_reason = map_core_finish_reason(result.finish)
                if finish_reason is None and stop_sequences.check(text):
                    finish_reason = FinishReason.STOP

                token_logprobs = _build_token_logprobs(
                    tokenizer,
                    logprobs,
                    token_id,
                    top_n=options.top_logprobs,
                )

                yield TokenEvent(
                    token_id=token_id,
                    text=text,
                    finish_reason=finish_reason,
                    logprobs=token_logprobs,
                    prompt_tokens=prompt_token_count,
                    generation_tokens=result.generation_tokens,
                )

                if finish_reason is not None:
                    return

                assert next_logits is not None
        elif use_processor_prepared_step_path:
            prepared = prepare_decode_step(
                logits,
                execution=execution,
                select_token=lambda step_logits: _select_token_from_logits(sampler, step_logits),
            )

            while True:
                next_raw_logits = None
                if len(generated_tokens) + 1 < options.max_tokens:
                    next_raw_logits = prepare_next_logits(
                        model,
                        state,
                        prepared,
                        execution=execution,
                        step_fn=step_fn,
                        prime_logits=True,
                    )

                result = materialize_prepared_step(
                    state,
                    prepared,
                    termination=termination,
                    execution=execution,
                    force_eval=not generated_tokens,
                )

                token_id = result.token_id
                generated_tokens.append(token_id)
                text = tokenizer.decode(token_id)
                finish_reason = map_core_finish_reason(result.finish)
                if finish_reason is None and stop_sequences.check(text):
                    finish_reason = FinishReason.STOP

                yield TokenEvent(
                    token_id=token_id,
                    text=text,
                    finish_reason=finish_reason,
                    logprobs=None,
                    prompt_tokens=prompt_token_count,
                    generation_tokens=result.generation_tokens,
                )

                if finish_reason is not None:
                    return

                assert next_raw_logits is not None
                step_logits = next_raw_logits.logits
                token_history = mx.array(generated_tokens, dtype=mx.int32)
                for processor in logits_processors:
                    step_logits = processor(token_history, step_logits)
                prepared = prepare_decode_step(
                    step_logits,
                    execution=execution,
                    select_token=lambda next_step_logits: _select_token_from_logits(
                        sampler,
                        next_step_logits,
                    ),
                )
        else:
            while True:
                step_logits = logits
                if logits_processors:
                    token_history = (
                        mx.array(generated_tokens, dtype=mx.int32)
                        if generated_tokens
                        else mx.array([], dtype=mx.int32)
                    )
                    for processor in logits_processors:
                        step_logits = processor(token_history, step_logits)

                logprobs = step_logits - mx.logsumexp(step_logits, axis=-1, keepdims=True)
                token_logprobs = None

                result, next_logits = decode_step(
                    model,
                    state,
                    step_logits,
                    termination=termination,
                    execution=execution,
                    select_token=lambda step_logits: _select_token_from_logits(
                        sampler,
                        step_logits,
                    ),
                    step_fn=step_fn,
                )

                token_id = result.token_id
                generated_tokens.append(token_id)
                text = tokenizer.decode(token_id)
                finish_reason = map_core_finish_reason(result.finish)
                if finish_reason is None and stop_sequences.check(text):
                    finish_reason = FinishReason.STOP

                if options.logprobs:
                    token_logprobs = _build_token_logprobs(
                        tokenizer,
                        logprobs,
                        token_id,
                        top_n=options.top_logprobs,
                    )

                yield TokenEvent(
                    token_id=token_id,
                    text=text,
                    finish_reason=finish_reason,
                    logprobs=token_logprobs,
                    prompt_tokens=prompt_token_count,
                    generation_tokens=result.generation_tokens,
                )

                if finish_reason is not None:
                    return

                assert next_logits is not None
                logits = next_logits
    finally:
        if final_cache_out is not None:
            final_cache_out.append(state.export_cache())
