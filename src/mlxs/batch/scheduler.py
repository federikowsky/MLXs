"""Batch scheduler — request management and continuous batching (§6.4, AC2).

Manages concurrent generation sequences. Handles enqueue, prefill,
decode, and removal of sequences. Model-agnostic — operates through
ModelProtocol.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import FinishReason, GenerateOptions, TokenEvent
from mlxs.cache.kv import KVCache
from mlxs.generate.logits import make_logits_processors
from mlxs.generate.sampling import make_sampler
from mlxs.generate.stop import StopCondition
from mlxs.protocols.generate import TokenizerProtocol

logger = logging.getLogger(__name__)


class _SeqState(Enum):
    """Lifecycle state of a sequence in the batch."""

    PENDING = auto()  # Waiting to be prefilled
    PREFILLING = auto()  # Currently in prefill
    DECODING = auto()  # Actively decoding
    FINISHED = auto()  # Generation complete


@dataclass
class _Sequence:
    """Internal state for one sequence in the batch."""

    request_id: str
    model: nn.Module
    tokenizer: TokenizerProtocol
    prompt_tokens: list[int]
    options: GenerateOptions
    state: _SeqState = _SeqState.PENDING

    # Resolved once before decode (O2)
    sampler: Any = None
    stop: StopCondition | None = None
    logits_processors: list | None = None

    # Runtime state
    cache: list[KVCache] | None = None
    current_token: mx.array | None = None
    events: list[TokenEvent] = field(default_factory=list)
    prompt_token_count: int = 0

    def setup(self) -> None:
        """Resolve abstractions once before decode loop (O2)."""
        self.sampler = make_sampler(
            temperature=self.options.temperature,
            top_p=self.options.top_p,
            top_k=self.options.top_k,
            min_p=self.options.min_p,
        )
        self.stop = StopCondition(
            eos_token_id=self.tokenizer.eos_token_id,
            max_tokens=self.options.max_tokens,
            stop_sequences=self.options.stop_sequences,
            extra_eos_token_ids=self.options.extra_eos_token_ids,
        )
        self.logits_processors = make_logits_processors(
            repetition_penalty=self.options.repetition_penalty,
        )
        self.prompt_token_count = len(self.prompt_tokens)


class BatchScheduler:
    """Continuous batching scheduler (§6.4, AC2).

    Manages concurrent generation sequences. Supports adding/removing
    sequences mid-batch. Sequences go through:
    PENDING → PREFILLING → DECODING → FINISHED.

    Satisfies ``BatchSchedulerProtocol``.
    """

    __slots__ = (
        "_active",
        "_completion_batch_size",
        "_finished",
        "_pending",
        "_prefill_step_size",
    )

    def __init__(
        self,
        *,
        completion_batch_size: int = 4,
        prefill_step_size: int = 2048,
    ) -> None:
        self._completion_batch_size = completion_batch_size
        self._prefill_step_size = prefill_step_size
        self._pending: OrderedDict[str, _Sequence] = OrderedDict()
        self._active: OrderedDict[str, _Sequence] = OrderedDict()
        self._finished: OrderedDict[str, _Sequence] = OrderedDict()

    def add(
        self,
        request_id: str,
        model: nn.Module,
        tokenizer: TokenizerProtocol,
        prompt: str | list[int],
        options: GenerateOptions,
    ) -> None:
        """Enqueue a new generation request."""
        prompt_tokens = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        seq = _Sequence(
            request_id=request_id,
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            options=options,
        )
        self._pending[request_id] = seq
        logger.debug("Added request %s (prompt_len=%d)", request_id, len(prompt_tokens))

    def remove(self, request_id: str) -> None:
        """Cancel and remove a request."""
        if request_id in self._pending:
            del self._pending[request_id]
        elif request_id in self._active:
            seq = self._active.pop(request_id)
            # Mark final event as cancelled
            if seq.events:
                seq.events[-1].finish_reason = FinishReason.CANCELLED
            seq.state = _SeqState.FINISHED
            self._finished[request_id] = seq
        # If already finished, no-op

    def step(self) -> dict[str, list[TokenEvent]]:
        """Run one batch step for all active + pending sequences.

        Returns:
            Mapping of request_id → new TokenEvents from this step.
        """
        results: dict[str, list[TokenEvent]] = {}

        # Prefill pending sequences (one at a time for simplicity)
        self._prefill_pending(results)

        # Decode active sequences
        self._decode_active(results)

        return results

    def _prefill_pending(self, results: dict[str, list[TokenEvent]]) -> None:
        """Prefill one pending sequence if there's capacity."""
        if not self._pending:
            return
        if len(self._active) >= self._completion_batch_size:
            return

        # Take the first pending sequence
        request_id, seq = next(iter(self._pending.items()))
        del self._pending[request_id]

        seq.state = _SeqState.PREFILLING
        seq.setup()
        seq.cache = seq.model.make_cache()

        prompt_array = mx.array(seq.prompt_tokens)
        total = len(seq.prompt_tokens)

        # Chunked prefill
        offset = 0
        while total - offset > 1:
            remaining = (total - offset) - 1
            n = min(self._prefill_step_size, remaining)
            chunk = prompt_array[offset : offset + n]
            seq.model(chunk[None], cache=seq.cache)
            mx.eval([c.state for c in seq.cache if c.state is not None])
            offset += n
            mx.clear_cache()

        # Last token — get logits and sample first token
        last_token = prompt_array[offset:]
        logits = seq.model(last_token[None], cache=seq.cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        y = seq.sampler(logprobs)
        mx.eval(y)

        token_id = y.item()
        text = seq.tokenizer.decode(token_id)
        finish_reason = seq.stop.check(token_id, text)

        event = TokenEvent(
            token_id=token_id,
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=seq.prompt_token_count,
            generation_tokens=1,
        )
        seq.events.append(event)
        results[request_id] = [event]

        if finish_reason is not None:
            seq.state = _SeqState.FINISHED
            self._finished[request_id] = seq
        else:
            seq.current_token = y
            seq.state = _SeqState.DECODING
            self._active[request_id] = seq

    def _decode_active(self, results: dict[str, list[TokenEvent]]) -> None:
        """Run one decode step for each active sequence."""
        finished_ids: list[str] = []

        for request_id, seq in self._active.items():
            if seq.current_token is None:
                continue

            logits = seq.model(seq.current_token[None], cache=seq.cache)
            logits = logits[:, -1, :]

            # Apply logits processors
            if seq.logits_processors:
                gen_tokens = [e.token_id for e in seq.events]
                all_tokens = mx.array(gen_tokens) if gen_tokens else mx.array([], dtype=mx.int32)
                for processor in seq.logits_processors:
                    logits = processor(all_tokens, logits)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = seq.sampler(logprobs)
            mx.eval(y)

            token_id = y.item()
            text = seq.tokenizer.decode(token_id)
            gen_count = len(seq.events) + 1
            finish_reason = seq.stop.check(token_id, text)

            event = TokenEvent(
                token_id=token_id,
                text=text,
                finish_reason=finish_reason,
                prompt_tokens=seq.prompt_token_count,
                generation_tokens=gen_count,
            )
            seq.events.append(event)
            results.setdefault(request_id, []).append(event)

            if finish_reason is not None:
                seq.state = _SeqState.FINISHED
                finished_ids.append(request_id)
            else:
                seq.current_token = y

        # Move finished sequences out of active
        for rid in finished_ids:
            seq = self._active.pop(rid)
            self._finished[rid] = seq

    def drain(self) -> Iterator[tuple[str, list[TokenEvent]]]:
        """Drain all finished sequences."""
        while self._finished:
            request_id, seq = self._finished.popitem(last=False)
            yield request_id, seq.events

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def finished_count(self) -> int:
        return len(self._finished)
