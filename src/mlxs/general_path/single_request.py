"""Layer 2 single-request rich generation semantics above Layer 1."""

from __future__ import annotations

from collections.abc import Iterator
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
from mlxs.runtime_core.decode import decode_step
from mlxs.runtime_core.prefill import run_prefill


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

    top_indices = mx.argpartition(logprobs_row, kth=-top_n)[-top_n:]
    top_indices = top_indices[mx.argsort(logprobs_row[top_indices])[::-1]]
    mx.eval(top_indices)

    return tuple(
        TopLogprob(
            token_id=int(idx.item()),
            token=tokenizer.decode(int(idx.item())),
            logprob=float(logprobs_row[int(idx.item())].item()),
        )
        for idx in top_indices
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
                select_token=sampler,
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
