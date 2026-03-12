"""Sampling strategies — temperature, top_p, top_k, min_p (§6.1, FR9).

Builds a sampler chain from GenerateOptions. Each filter operates on
log-probabilities and returns filtered log-probabilities. The final step
samples from the filtered distribution via categorical sampling.

All compiled functions use mx.compile with random state tracking.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import mlx.core as mx

# -- Compiled sampling primitives -------------------------------------------


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logprobs: mx.array, temp: float) -> mx.array:
    """Sample from logprobs with temperature scaling."""
    return mx.random.categorical(logprobs * (1 / temp))


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """Nucleus sampling — keep tokens whose cumulative probability exceeds 1-top_p."""
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)
    return mx.where(cumulative_probs > 1 - top_p, logprobs, -float("inf"))


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """Keep only top-k tokens by probability."""
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    return mx.put_along_axis(logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    """Min-p sampling — keep tokens above min_p * max_prob."""
    sorted_indices = mx.argsort(-logprobs, axis=-1)
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)

    import math

    scaled_min_p = sorted_logprobs[:, 0:1] + math.log(min_p)
    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False
    selected_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)

    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    return mx.take_along_axis(selected_logprobs, inverse_indices, axis=-1)


# -- Sampler factory --------------------------------------------------------

SamplerFn = Callable[[mx.array], mx.array]


def make_sampler(
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
) -> SamplerFn:
    """Build a sampler function from generation options.

    Args:
        temperature: Sampling temperature. 0 = greedy (argmax).
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        top_k: Top-k filtering. 0 = disabled.
        min_p: Min-p threshold. 0.0 = disabled.

    Returns:
        A callable that takes log-probabilities and returns sampled token ids.
    """
    if temperature == 0:
        return lambda x: mx.argmax(x, axis=-1)

    chain: list[Callable[[mx.array], mx.array]] = []
    if 0 < top_p < 1.0:
        chain.append(lambda x: apply_top_p(x, top_p))
    if min_p > 0:
        chain.append(lambda x: apply_min_p(x, min_p))
    if top_k > 0:
        chain.append(lambda x: apply_top_k(x, top_k))

    def sampler(logprobs: mx.array) -> mx.array:
        for fn in chain:
            logprobs = fn(logprobs)
        return categorical_sampling(logprobs, temperature)

    return sampler
