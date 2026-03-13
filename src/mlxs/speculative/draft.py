"""Draft model token generation for speculative decoding (§6.5).

Generates N candidate tokens from a smaller/faster draft model.
These tokens are then verified by the target model in verify.py.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache


def draft_tokens(
    draft_model: nn.Module,
    draft_cache: list[KVCache],
    last_token: mx.array,
    *,
    num_tokens: int,
    sampler: Callable[[mx.array], mx.array],
) -> tuple[mx.array, mx.array]:
    """Generate draft tokens from the draft model.

    Args:
        draft_model: The smaller/faster draft model.
        draft_cache: KV cache for the draft model.
        last_token: Last accepted token, shape (1,).
        num_tokens: Number of draft tokens to generate.
        sampler: Sampling function (logprobs → token id).

    Returns:
        Tuple of (draft_token_ids, draft_logprobs):
        - draft_token_ids: shape (num_tokens,)
        - draft_logprobs: shape (num_tokens, vocab_size) — log probabilities
          for each draft position.
    """
    tokens: list[mx.array] = []
    all_logprobs: list[mx.array] = []
    current = last_token

    for _ in range(num_tokens):
        logits = draft_model(current[None], cache=draft_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        token = sampler(logprobs)
        mx.eval(token)

        tokens.append(token)
        all_logprobs.append(logprobs.squeeze(0))
        current = token

    return mx.concatenate(tokens), mx.stack(all_logprobs)
