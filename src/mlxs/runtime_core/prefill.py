"""Layer 1 prefill implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import mlx.core as mx

from mlxs.protocols.model import ModelProtocol
from mlxs.runtime_core.policy import CoreExecutionPolicy, stream_context
from mlxs.runtime_core.state import CoreState


def _as_prompt_array(prompt_tokens: Sequence[int] | mx.array) -> mx.array:
    if hasattr(prompt_tokens, "ndim"):
        return cast(mx.array, prompt_tokens)
    return mx.array(list(prompt_tokens))


def _validate_input_embeddings(
    prompt_array: mx.array,
    input_embeddings: mx.array | None,
) -> None:
    if input_embeddings is None:
        return
    if input_embeddings.ndim != 2:
        raise ValueError(
            f"input_embeddings must be 2-D (T, D), got shape {input_embeddings.shape}"
        )
    if input_embeddings.shape[0] != len(prompt_array):
        raise ValueError(
            "input_embeddings length "
            f"({input_embeddings.shape[0]}) must match prompt length ({len(prompt_array)})"
        )


def run_prefill(
    model: ModelProtocol,
    prompt_tokens: Sequence[int] | mx.array,
    state: CoreState,
    *,
    execution: CoreExecutionPolicy,
    prefill_step_size: int = 2048,
    input_embeddings: mx.array | None = None,
) -> mx.array:
    """Process prompt tokens into active runtime state and return first-step logits."""
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be >= 1")

    prompt_array = _as_prompt_array(prompt_tokens)
    if len(prompt_array) == 0:
        raise ValueError("Prompt must not be empty")

    _validate_input_embeddings(prompt_array, input_embeddings)
    state.record_prompt(len(prompt_array))

    total = len(prompt_array)
    offset = 0
    while total - offset > 1:
        remaining = (total - offset) - 1
        n_tokens = min(prefill_step_size, remaining)
        chunk = prompt_array[offset : offset + n_tokens]
        with stream_context(execution):
            if input_embeddings is not None:
                chunk_embeddings = input_embeddings[offset : offset + n_tokens]
                model(chunk[None], cache=state.cache, input_embeddings=chunk_embeddings[None])
            else:
                model(chunk[None], cache=state.cache)

            cache_states = [
                cache_state
                for cache_state in (getattr(cache, "state", None) for cache in state.cache)
                if cache_state is not None
            ]
            if cache_states:
                mx.eval(cache_states)
        offset += n_tokens
        mx.clear_cache()

    last_token = prompt_array[offset:]
    with stream_context(execution):
        if input_embeddings is not None:
            last_embeddings = input_embeddings[offset:]
            logits = model(
                last_token[None],
                cache=state.cache,
                input_embeddings=last_embeddings[None],
            )
        else:
            logits = model(last_token[None], cache=state.cache)
    return logits[:, -1, :]
