"""Generate module — single-request inference (§6.1, §9, FR3).

Public API: ``generate()`` — takes model, tokenizer, prompt, options
and yields a stream of TokenEvent objects.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._errors import InvalidPromptError
from mlxs._types import GenerateOptions, TokenEvent
from mlxs.adaptive_kv import (
    AdaptiveKVCompatibilityError,
    AdaptiveKVConfig,
    AdaptiveKVManager,
    assess_generation_compatibility,
)
from mlxs.adaptive_kv.metrics import emit_compatibility_fallback
from mlxs.cache.kv import KVCache
from mlxs.generate.decode import decode_loop
from mlxs.generate.logits import make_logits_processors
from mlxs.generate.prefill import chunked_prefill
from mlxs.generate.sampling import make_sampler
from mlxs.generate.stop import StopCondition
from mlxs.observability.metrics import NoOpMetrics
from mlxs.protocols.generate import TokenizerProtocol
from mlxs.protocols.metrics import MetricsProtocol


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
    adaptive_config: AdaptiveKVConfig | None = None,
    metrics: MetricsProtocol | None = None,
    final_adaptive_state_out: list[dict[str, Any]] | None = None,
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
        input_embeddings: Pre-computed embeddings ``(T, D)`` from
            multimodal preprocessing (§7.4). When provided, used instead
            of ``embed_tokens`` during prefill.
        prefill_step_size: Max tokens per prefill chunk.
        compile_decode: If True, compile the model forward for decode (§6.8).
            Falls back to uncompiled on failure (AC12).
        clear_cache_interval: Steps between mx.clear_cache() calls.
            0 = disabled. Default: 256.
        final_cache_out: If provided, the list is appended with the KV cache
            after generation completes (for prompt_cache.put). Plan-chat-cli.

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

    # Validate input_embeddings shape (§7.4)
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

    # Set seed if specified
    if options.seed is not None:
        mx.random.seed(options.seed)

    metrics_sink = metrics if metrics is not None else NoOpMetrics()
    adaptive_manager: AdaptiveKVManager | None = None
    adaptive_enabled = adaptive_config is not None and adaptive_config.enabled

    # Create KV cache if not provided
    if adaptive_enabled:
        compatibility = assess_generation_compatibility(
            model,
            cache=cache,
            compile_decode=compile_decode,
            quantized_kv_start=quantized_kv_start,
            input_embeddings_present=input_embeddings is not None,
        )
        if not compatibility.supported:
            reason = compatibility.reason or "adaptive_kv_v1 unsupported"
            emit_compatibility_fallback(metrics_sink, reason=reason)
            if final_adaptive_state_out is not None:
                final_adaptive_state_out.append({"enabled": False, "reason": reason})
            raise AdaptiveKVCompatibilityError(reason)
        adaptive_manager = AdaptiveKVManager(
            adaptive_config,
            num_layers=compatibility.num_layers,
            metrics=metrics_sink,
        )
        adaptive_manager.bind_generation_context(
            model=model,
            prefill_step_size=prefill_step_size,
        )
        cache = adaptive_manager.caches()
    elif cache is None:
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

    # Build compiled forward if requested (§6.8, AC12 fallback-safe)
    forward_fn = None
    if compile_decode:
        try:
            from mlxs.generate.compile import make_compiled_step

            forward_fn = make_compiled_step(model)
        except Exception:
            # Fallback to uncompiled (AC12)
            forward_fn = None

    # Prefill: process prompt through model
    first_logits = chunked_prefill(
        model,
        prompt_array,
        cache,
        prefill_step_size=prefill_step_size,
        input_embeddings=input_embeddings,
        adaptive_manager=adaptive_manager,
    )

    def _gen() -> Iterator[TokenEvent]:
        try:
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
                forward_fn=forward_fn,
                clear_cache_interval=clear_cache_interval,
                quantized_kv_start=quantized_kv_start,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                adaptive_manager=adaptive_manager,
            )
        finally:
            if final_cache_out is not None and adaptive_manager is None:
                final_cache_out.append(cache)
            if final_adaptive_state_out is not None and adaptive_manager is not None:
                final_adaptive_state_out.append(adaptive_manager.debug_snapshot())

    return _gen()


__all__ = ["generate"]
