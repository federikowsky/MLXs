"""Chunked prefill — process long prompts in chunks (§6.1, FR3).

Processes the prompt through the model in fixed-size chunks to bound
memory usage during prefill. Calls mx.eval after each chunk to free
intermediate buffers.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.adaptive_kv.manager import AdaptiveKVManager
from mlxs.cache.kv import KVCache


def chunked_prefill(
    model: nn.Module,
    prompt_tokens: mx.array,
    cache: list[KVCache] | list[Any],
    *,
    prefill_step_size: int = 2048,
    input_embeddings: mx.array | None = None,
    adaptive_manager: AdaptiveKVManager | None = None,
) -> mx.array:
    """Run prefill on a prompt, processing in chunks.

    Processes all tokens except the last one in chunks, then returns
    the logits from the final token (which becomes the first decode step).

    Args:
        model: The model to run.
        prompt_tokens: 1-D token array (not batched).
        cache: KV cache list (one per layer).
        prefill_step_size: Maximum tokens per prefill chunk.
        input_embeddings: Pre-computed embeddings ``(T, D)`` from
            multimodal preprocessing (§7.4). Sliced in sync with tokens.

    Returns:
        Logits array of shape (1, vocab_size) from the last prompt token.
    """
    if adaptive_manager is not None and adaptive_manager.prompt_token_count == 0:
        adaptive_manager.initialize_prompt([int(token.item()) for token in prompt_tokens])

    total = len(prompt_tokens)

    # Process all tokens except the last one in chunks
    offset = 0
    while total - offset > 1:
        remaining = (total - offset) - 1
        n = min(prefill_step_size, remaining)
        chunk = prompt_tokens[offset : offset + n]
        if input_embeddings is not None:
            chunk_embeds = input_embeddings[offset : offset + n]
            model(chunk[None], cache=cache, input_embeddings=chunk_embeds[None])
        else:
            model(chunk[None], cache=cache)
        mx.eval([c.state for c in cache if c.state is not None])
        offset += n
        mx.clear_cache()

    # Process the last token and return its logits
    last_token = prompt_tokens[offset:]
    if input_embeddings is not None:
        last_embed = input_embeddings[offset:]
        logits = model(last_token[None], cache=cache, input_embeddings=last_embed[None])
    else:
        logits = model(last_token[None], cache=cache)
    return logits[:, -1, :]
