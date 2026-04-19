"""Private Layer 1 batched progression for the canonical AC2 fast path."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx

from mlxs.cache.kv import KVCache
from mlxs.protocols.model import ModelProtocol
from mlxs.runtime_core.policy import CoreExecutionPolicy, stream_context
from mlxs.runtime_core.selection import TokenSelector, greedy_select

BatchStepFn = Callable[[mx.array], mx.array]


@dataclass(slots=True)
class BatchCoreState:
    """Layer 1 owner of active mutable batch cache state."""

    cache: list[KVCache]

    @classmethod
    def create(cls, model: ModelProtocol) -> BatchCoreState:
        cache = model.make_cache()
        if not cache or not all(type(layer) is KVCache for layer in cache):
            raise ValueError("batch progression only supports plain KVCache in the canonical fast path")
        return cls(cache=cache)

    def filter_rows(self, keep_idx: Sequence[int]) -> None:
        if not keep_idx:
            raise ValueError("keep_idx must not be empty")
        keep = mx.array(list(keep_idx), dtype=mx.int32)
        for layer in self.cache:
            keys, values = layer.state
            layer.state = (keys[keep], values[keep])

    def extract_row(self, idx: int) -> list[KVCache]:
        extracted: list[KVCache] = []
        for layer in self.cache:
            keys, values = layer.state
            out = KVCache()
            out.state = (
                keys[idx : idx + 1],
                values[idx : idx + 1],
            )
            extracted.append(out)
        return extracted


@dataclass(frozen=True, slots=True)
class PreparedBatchDecodeStep:
    """One prepared batch token step owned by Layer 1."""

    logits: mx.array
    tokens: mx.array


def _default_batch_step_fn(
    model: ModelProtocol,
    state: BatchCoreState,
) -> BatchStepFn:
    def step(input_ids: mx.array) -> mx.array:
        return model(input_ids, cache=state.cache)

    return step


def _resolve_batch_step_fn(
    model: ModelProtocol,
    state: BatchCoreState,
    step_fn: BatchStepFn | None,
) -> BatchStepFn:
    return step_fn if step_fn is not None else _default_batch_step_fn(model, state)


def run_batch_prefill(
    model: ModelProtocol,
    prompt_batch: mx.array,
    state: BatchCoreState,
    *,
    execution: CoreExecutionPolicy,
    prefill_step_size: int = 2048,
    step_fn: BatchStepFn | None = None,
) -> mx.array:
    """Process an aligned batch prompt into active Layer 1 batch state."""
    if prompt_batch.ndim != 2:
        raise ValueError(f"prompt_batch must be 2-D (B, T), got {prompt_batch.shape}")
    if prompt_batch.shape[1] < 1:
        raise ValueError("prompt_batch must contain at least one token")
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be >= 1")

    resolved_step = _resolve_batch_step_fn(model, state, step_fn)
    total = int(prompt_batch.shape[1])
    offset = 0
    while total - offset > 1:
        remaining = (total - offset) - 1
        n_tokens = min(prefill_step_size, remaining)
        chunk = prompt_batch[:, offset : offset + n_tokens]
        with stream_context(execution):
            resolved_step(chunk)
            cache_states = [
                cache_state
                for cache_state in (getattr(cache, "state", None) for cache in state.cache)
                if cache_state is not None
            ]
            if cache_states:
                mx.eval(cache_states)
        offset += n_tokens
        mx.clear_cache()

    last_token = prompt_batch[:, offset:]
    with stream_context(execution):
        logits = resolved_step(last_token)
    return logits[:, -1, :]


def prepare_batch_decode_step(
    logits: mx.array,
    *,
    execution: CoreExecutionPolicy,
    select_token: TokenSelector = greedy_select,
    prime_tokens: bool = True,
) -> PreparedBatchDecodeStep:
    """Prepare the current batch token under Layer 1 execution discipline."""
    with stream_context(execution):
        tokens = select_token(logits)
        if prime_tokens:
            mx.async_eval(tokens)
    return PreparedBatchDecodeStep(logits=logits, tokens=tokens)


def schedule_next_batch_decode_step(
    model: ModelProtocol,
    state: BatchCoreState,
    prepared: PreparedBatchDecodeStep,
    *,
    execution: CoreExecutionPolicy,
    step_fn: BatchStepFn | None = None,
    select_token: TokenSelector = greedy_select,
    prime_tokens: bool = True,
) -> PreparedBatchDecodeStep:
    """Advance the batched cache by one token and prepare the next batch token."""
    resolved_step = _resolve_batch_step_fn(model, state, step_fn)
    with stream_context(execution):
        next_logits = resolved_step(prepared.tokens[:, None])[:, -1, :]
        next_tokens = select_token(next_logits)
        if prime_tokens:
            mx.async_eval(next_tokens)
    return PreparedBatchDecodeStep(logits=next_logits, tokens=next_tokens)


def materialize_prepared_batch_step(
    prepared: PreparedBatchDecodeStep,
    *,
    execution: CoreExecutionPolicy,
    next_prepared: PreparedBatchDecodeStep | None = None,
    force_eval: bool = False,
) -> tuple[int, ...]:
    """Materialize a prepared batch step as token ids."""
    del execution
    if next_prepared is not None:
        mx.eval(prepared.tokens, next_prepared.tokens)
    elif force_eval:
        mx.eval(prepared.tokens)
    return tuple(int(token_id) for token_id in prepared.tokens.tolist())


def filter_prepared_batch_step(
    prepared: PreparedBatchDecodeStep,
    keep_idx: Sequence[int],
) -> PreparedBatchDecodeStep:
    """Filter a prepared batch step down to the kept rows."""
    keep = mx.array(list(keep_idx), dtype=mx.int32)
    return PreparedBatchDecodeStep(
        logits=prepared.logits[keep],
        tokens=prepared.tokens[keep],
    )
