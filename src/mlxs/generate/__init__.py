"""Generate module — single-request inference (§6.1, §9, FR3).

Public API: ``generate()`` — takes model, tokenizer, prompt, options
and yields a stream of TokenEvent objects.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import mlx.core as mx

from mlxs._errors import InvalidPromptError
from mlxs._types import GenerateOptions, TokenEvent
from mlxs.generate.capabilities import resolve_decode_capabilities
from mlxs.generate.decode import (
    decode_async_eval_enabled,
    decode_loop,
    decode_profile_enabled,
    emit_decode_profile_report,
    prepare_decode_plan,
)
from mlxs.generate.prefill import chunked_prefill
from mlxs.protocols.generate import TokenizerProtocol


def generate(
    model: Any,
    tokenizer: TokenizerProtocol,
    prompt: str | list[int],
    options: GenerateOptions | None = None,
    *,
    cache: list[Any] | None = None,
    input_embeddings: mx.array | None = None,
    prefill_step_size: int = 2048,
    compile_decode: bool = False,
    clear_cache_interval: int = 256,
    quantized_kv_start: int = 0,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    final_cache_out: list[list[Any]] | None = None,
) -> Iterator[TokenEvent]:
    """Generate tokens from a prompt (§6.1, FR3)."""
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

    capabilities = resolve_decode_capabilities(model, cache)

    if input_embeddings is not None and not capabilities.supports_prefill_input_embeddings:
        raise InvalidPromptError("Model does not support input_embeddings during prefill")

    if (
        quantized_kv_start > 0
        and kv_bits is not None
        and not capabilities.supports_delayed_kv_quantization
    ):
        raise ValueError("Delayed quantized KV requires full-precision KV caches")

    first_logits = chunked_prefill(
        model,
        prompt_array,
        cache,
        prefill_step_size=prefill_step_size,
        input_embeddings=input_embeddings,
        capabilities=capabilities,
    )

    async_eval = decode_async_eval_enabled()
    plan = prepare_decode_plan(
        model,
        cache,
        options=options,
        decoder=tokenizer.decode,
        eos_token_id=tokenizer.eos_token_id,
        prompt_token_count=prompt_token_count,
        capabilities=capabilities,
        compile_decode=compile_decode,
        async_eval=async_eval,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
    )

    profile: dict[str, Any] | None = None
    if decode_profile_enabled():
        profile = {
            "forward_decode_s": 0.0,
            "logits_sample_prep_s": 0.0,
            "sync_enqueue_s": 0.0,
            "sync_wait_token_s": 0.0,
            "sync_wait_event_s": 0.0,
            "token_boundary_mode": plan.sync.token_boundary.name,
            "token_boundary_selection": plan.sync.token_boundary.selection,
            "token_boundary_reason": plan.sync.token_boundary.reason,
            "mx_async_eval_s": 0.0,
            "mx_eval_s": 0.0,
            "materialize_s": 0.0,
            "mutation_s": 0.0,
            "n_forward_decode": 0,
            "sync_enqueue_calls": 0,
            "sync_enqueue_tensors": 0,
            "sync_wait_token_calls": 0,
            "sync_wait_token_tensors": 0,
            "sync_wait_event_calls": 0,
            "sync_wait_event_tensors": 0,
            "token_boundary_steps": 0,
            "token_boundary_wait_reuses_enqueue_steps": 0,
            "token_boundary_event_wait_empty_steps": 0,
            "token_boundary_token_only_steps": 0,
            "forward_wall_samples": [],
            "step_wall_samples": [],
        }

    def _gen() -> Iterator[TokenEvent]:
        try:
            yield from decode_loop(
                cache,
                first_logits,
                plan=plan,
                clear_cache_interval=clear_cache_interval,
                quantized_kv_start=quantized_kv_start,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                profile=profile,
            )
        finally:
            if final_cache_out is not None:
                final_cache_out.append(cache)
            if profile is not None:
                emit_decode_profile_report(
                    profile,
                    compile_decode=compile_decode,
                    async_eval=async_eval,
                )

    return _gen()


__all__ = ["generate"]
