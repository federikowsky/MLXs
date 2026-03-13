"""Verification and accept/reject logic for speculative decoding (§6.5).

Verifies draft tokens against the target model and determines
which tokens to accept. On rejection, rewinds both draft and
target KV caches to the last accepted position.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import GenerateOptions, TokenEvent
from mlxs.cache.kv import KVCache
from mlxs.generate.prefill import chunked_prefill
from mlxs.generate.sampling import make_sampler
from mlxs.generate.stop import StopCondition
from mlxs.speculative.draft import draft_tokens


def _verify_and_accept(
    target_logprobs: mx.array,
    draft_logprobs: mx.array,
    draft_tokens_arr: mx.array,
    sampler: Callable[[mx.array], mx.array],
) -> tuple[mx.array, int]:
    """Verify draft tokens against target model probabilities.

    Uses rejection sampling: accept draft token if target probability
    >= draft probability, otherwise reject with probability proportional
    to the ratio.

    Args:
        target_logprobs: Target model log-probs at each draft position,
            shape (num_draft + 1, vocab).
        draft_logprobs: Draft model log-probs, shape (num_draft, vocab).
        draft_tokens_arr: Draft token IDs, shape (num_draft,).
        sampler: Sampling function for resampling on rejection.

    Returns:
        Tuple of (accepted_tokens, num_accepted):
        - accepted_tokens includes all accepted draft tokens plus one
          newly sampled token (either resampled on rejection or sampled
          from the position after all accepted drafts).
        - num_accepted: count of accepted draft tokens (0 to num_draft).
    """
    num_draft = draft_tokens_arr.shape[0]
    accepted: list[mx.array] = []

    for i in range(num_draft):
        draft_token = draft_tokens_arr[i]
        target_p = mx.exp(target_logprobs[i])
        draft_p = mx.exp(draft_logprobs[i])

        # Acceptance probability for this token
        target_prob = target_p[draft_token]
        draft_prob = draft_p[draft_token]

        # Accept if target prob >= draft prob
        r = mx.random.uniform()
        mx.eval(r, target_prob, draft_prob)

        if target_prob.item() >= draft_prob.item() or r.item() < (
            target_prob.item() / max(draft_prob.item(), 1e-10)
        ):
            accepted.append(draft_token)
        else:
            # Reject: resample from adjusted distribution
            # p'(x) = max(0, p_target(x) - p_draft(x))
            adjusted = mx.maximum(target_p - draft_p, mx.array(0.0))
            adjusted_sum = mx.sum(adjusted)
            if adjusted_sum.item() > 0:
                adjusted_logprobs = mx.log(adjusted / adjusted_sum)
            else:
                adjusted_logprobs = target_logprobs[i]
            new_token = sampler(adjusted_logprobs)
            mx.eval(new_token)
            accepted.append(new_token)
            return mx.stack(accepted), len(accepted) - 1

    # All draft tokens accepted — sample next token from target
    next_token = sampler(target_logprobs[num_draft])
    mx.eval(next_token)
    accepted.append(next_token)
    return mx.stack(accepted), num_draft


def _rewind_cache(cache: list[KVCache], keep_tokens: int) -> None:
    """Rewind KV cache to a specific position.

    Truncates cache entries beyond the keep position.

    Args:
        cache: List of KV caches (one per layer).
        keep_tokens: Number of total tokens to keep in cache.
    """
    for layer_cache in cache:
        if hasattr(layer_cache, "rewind"):
            layer_cache.rewind(keep_tokens)
        elif layer_cache.offset > keep_tokens:
            # Fallback: trim keys/values directly
            trim = layer_cache.offset - keep_tokens
            if layer_cache.keys is not None:
                layer_cache.keys = layer_cache.keys[:, :, :-trim, :]
                layer_cache.values = layer_cache.values[:, :, :-trim, :]
                layer_cache.offset = keep_tokens


def speculative_generate(
    target_model: nn.Module,
    draft_model: nn.Module,
    tokenizer: object,
    prompt: str | list[int],
    options: GenerateOptions | None = None,
    *,
    num_draft_tokens: int = 5,
    prefill_step_size: int = 2048,
) -> Iterator[TokenEvent]:
    """Generate tokens using speculative decoding (§6.5, FR6).

    Uses a smaller draft model to propose candidate tokens, then
    verifies them in batch with the target model.

    Args:
        target_model: The main (larger) model.
        draft_model: The smaller/faster draft model. Must share tokenizer.
        tokenizer: Tokenizer with encode/decode/eos_token_id.
        prompt: Input text or pre-tokenized token ids.
        options: Generation parameters.
        num_draft_tokens: Draft tokens per verification step.
        prefill_step_size: Max tokens per prefill chunk.

    Yields:
        TokenEvent for each accepted token.
    """
    if options is None:
        options = GenerateOptions()

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    if not prompt_tokens:
        raise ValueError("Prompt must not be empty")

    prompt_array = mx.array(prompt_tokens)
    prompt_token_count = len(prompt_tokens)

    if options.seed is not None:
        mx.random.seed(options.seed)

    # Create caches
    target_cache = target_model.make_cache()
    draft_cache = draft_model.make_cache()

    # Build sampler
    sampler = make_sampler(
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        min_p=options.min_p,
    )

    # Build stop condition
    stop = StopCondition(
        eos_token_id=tokenizer.eos_token_id,
        max_tokens=options.max_tokens,
        stop_sequences=options.stop_sequences,
        extra_eos_token_ids=getattr(options, "extra_eos_token_ids", ()),
    )

    # Prefill both models
    target_logits = chunked_prefill(
        target_model, prompt_array, target_cache, prefill_step_size=prefill_step_size
    )
    chunked_prefill(draft_model, prompt_array, draft_cache, prefill_step_size=prefill_step_size)

    # Sample first token from target
    target_logprobs = target_logits - mx.logsumexp(target_logits, keepdims=True)
    last_token = sampler(target_logprobs)
    mx.eval(last_token)

    tokens_generated = 0
    token_id = last_token.item()
    text = tokenizer.decode(token_id)
    finish = stop.check(token_id, text)

    yield TokenEvent(
        token_id=token_id,
        text=text,
        finish_reason=finish,
        prompt_tokens=prompt_token_count,
        generation_tokens=tokens_generated + 1,
    )
    tokens_generated += 1

    if finish is not None:
        return

    while tokens_generated < options.max_tokens:
        # Save cache positions for potential rewind
        target_offset = target_cache[0].offset if target_cache else 0

        # 1. Generate draft tokens
        remaining = options.max_tokens - tokens_generated
        n_draft = min(num_draft_tokens, remaining - 1)
        if n_draft <= 0:
            break

        draft_tokens_arr, draft_lps = draft_tokens(
            draft_model,
            draft_cache,
            last_token,
            num_tokens=n_draft,
            sampler=sampler,
        )

        # 2. Verify: run target model on all draft tokens at once
        verify_input = mx.concatenate([last_token[None], draft_tokens_arr[None, :]])
        verify_input = verify_input.reshape(1, -1)
        target_logits = target_model(verify_input, cache=target_cache)
        target_logits = target_logits[0]  # (n_draft + 1, vocab)
        target_lps = target_logits - mx.logsumexp(target_logits, axis=-1, keepdims=True)
        mx.eval(target_lps)

        # 3. Accept/reject
        accepted, n_accepted = _verify_and_accept(target_lps, draft_lps, draft_tokens_arr, sampler)

        # 4. Rewind caches on partial rejection
        if n_accepted < n_draft:
            rewind_to = target_offset + n_accepted + 1
            _rewind_cache(target_cache, rewind_to)
            _rewind_cache(draft_cache, rewind_to)

        # 5. Yield accepted tokens
        for i in range(len(accepted)):
            token_id = accepted[i].item()
            text = tokenizer.decode(token_id)
            tokens_generated += 1
            finish = stop.check(token_id, text)

            yield TokenEvent(
                token_id=token_id,
                text=text,
                finish_reason=finish,
                prompt_tokens=prompt_token_count,
                generation_tokens=tokens_generated,
            )

            if finish is not None:
                return

        last_token = accepted[-1]

    # Max tokens reached — yield final event with LENGTH finish
    if tokens_generated >= options.max_tokens:
        return
