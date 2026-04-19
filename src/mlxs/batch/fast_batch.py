"""Layer 3 row orchestration over shared Layer 1 batch progression."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from mlxs._types import FinishReason, TokenEvent
from mlxs.cache.kv import KVCache
from mlxs.protocols.generate import TokenizerProtocol
from mlxs.runtime_core.batch_progression import (
    BatchCoreState,
    PreparedBatchDecodeStep,
    filter_prepared_batch_step,
    materialize_prepared_batch_step,
    prepare_batch_decode_step,
    run_batch_prefill,
    schedule_next_batch_decode_step,
)
from mlxs.runtime_core.policy import CoreExecutionPolicy, CoreTerminationPolicy


@dataclass(slots=True)
class _FastBatchRow:
    request_id: str
    tokenizer: TokenizerProtocol
    termination: CoreTerminationPolicy
    prompt_token_count: int
    generation_tokens: int = 0
    events: list[TokenEvent] = field(default_factory=list)

    def append(self, token_id: int) -> TokenEvent:
        self.generation_tokens += 1
        text = self.tokenizer.decode(token_id)
        finish = self.termination.finish_for(
            token_id,
            generation_tokens=self.generation_tokens,
        )
        finish_reason = None
        if finish is not None:
            finish_reason = (
                FinishReason.STOP if finish.name == "STOP" else FinishReason.LENGTH
            )
        event = TokenEvent(
            token_id=token_id,
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=self.prompt_token_count,
            generation_tokens=self.generation_tokens,
        )
        self.events.append(event)
        return event

    def cancel(self) -> None:
        if self.events:
            self.events[-1].finish_reason = FinishReason.CANCELLED


@dataclass(slots=True)
class _SharedFastBatch:
    model: object
    execution: CoreExecutionPolicy
    state: BatchCoreState
    rows: list[_FastBatchRow]
    prepared: PreparedBatchDecodeStep
    first_step: bool = True

    @classmethod
    def from_aligned_cohort(
        cls,
        *,
        model: object,
        rows: list[_FastBatchRow],
        prompt_tokens: list[list[int]],
        execution: CoreExecutionPolicy,
        prefill_step_size: int,
    ) -> _SharedFastBatch:
        if len(rows) <= 1:
            raise ValueError("fast batch requires at least two rows")
        state = BatchCoreState.create(model)
        logits = run_batch_prefill(
            model,
            mx.array(prompt_tokens),
            state,
            execution=execution,
            prefill_step_size=prefill_step_size,
        )
        prepared = prepare_batch_decode_step(
            logits,
            execution=execution,
        )
        return cls(
            model=model,
            execution=execution,
            state=state,
            rows=rows,
            prepared=prepared,
            first_step=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def contains(self, request_id: str) -> bool:
        return any(row.request_id == request_id for row in self.rows)

    def cancel(
        self,
        request_id: str,
    ) -> tuple[str, list[TokenEvent], list[KVCache] | None] | None:
        for idx, row in enumerate(self.rows):
            if row.request_id != request_id:
                continue
            row.cancel()
            final_cache = self.state.extract_row(idx)
            keep_idx = [i for i in range(len(self.rows)) if i != idx]
            if keep_idx:
                self.state.filter_rows(keep_idx)
                self.rows = [self.rows[i] for i in keep_idx]
                self.prepared = filter_prepared_batch_step(self.prepared, keep_idx)
            else:
                self.rows = []
            return request_id, row.events, final_cache
        return None

    def step(
        self,
    ) -> tuple[dict[str, list[TokenEvent]], list[tuple[str, list[TokenEvent], list[KVCache] | None]]]:
        next_prepared = schedule_next_batch_decode_step(
            self.model,
            self.state,
            self.prepared,
            execution=self.execution,
        )
        token_ids = materialize_prepared_batch_step(
            self.prepared,
            execution=self.execution,
            next_prepared=next_prepared,
            force_eval=self.first_step,
        )
        self.first_step = False

        results: dict[str, list[TokenEvent]] = {}
        finished: list[tuple[str, list[TokenEvent], list[KVCache] | None]] = []
        keep_idx: list[int] = []
        for idx, (row, token_id) in enumerate(zip(self.rows, token_ids, strict=True)):
            event = row.append(token_id)
            results[row.request_id] = [event]
            if event.finish_reason is None:
                keep_idx.append(idx)
            else:
                finished.append((row.request_id, row.events, self.state.extract_row(idx)))

        if keep_idx:
            if len(keep_idx) != len(self.rows):
                self.state.filter_rows(keep_idx)
                self.rows = [self.rows[i] for i in keep_idx]
                self.prepared = filter_prepared_batch_step(next_prepared, keep_idx)
            else:
                self.prepared = next_prepared
        else:
            self.rows = []

        return results, finished
