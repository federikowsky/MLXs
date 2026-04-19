from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from mlxs.cache.kv import KVCache
from mlxs.runtime_core.batch_progression import (
    BatchCoreState,
    filter_prepared_batch_step,
    materialize_prepared_batch_step,
    prepare_batch_decode_step,
    run_batch_prefill,
    schedule_next_batch_decode_step,
)
from mlxs.runtime_core.policy import CoreExecutionPolicy


class _BatchModel:
    def __init__(self, outputs: list[mx.array]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[int, ...]] = []

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        input_embeddings: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        del input_embeddings, mask
        self.calls.append(tuple(int(v) for v in input_ids.shape))
        if cache is not None:
            batch, seq_len = input_ids.shape
            for layer in cache:
                keys = mx.zeros((batch, 1, seq_len, 4), dtype=mx.float32)
                values = mx.zeros((batch, 1, seq_len, 4), dtype=mx.float32)
                layer.update_and_fetch(keys, values)
        return self.outputs.pop(0)


class _NonKVModel:
    def make_cache(self) -> list[object]:
        return [object()]


def test_batch_core_state_create_rejects_non_kv_cache() -> None:
    with pytest.raises(ValueError):
        BatchCoreState.create(_NonKVModel())


def test_run_batch_prefill_and_schedule_next_step() -> None:
    model = _BatchModel(
        [
            mx.zeros((2, 2, 3), dtype=mx.float32),
            mx.array(
                [
                    [[0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0]],
                ],
                dtype=mx.float32,
            ),
            mx.array(
                [
                    [[0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0]],
                ],
                dtype=mx.float32,
            ),
        ]
    )
    state = BatchCoreState.create(model)
    logits = run_batch_prefill(
        model,
        mx.array([[10, 11, 12], [20, 21, 22]]),
        state,
        execution=CoreExecutionPolicy(),
        prefill_step_size=2,
    )

    prepared = prepare_batch_decode_step(logits, execution=CoreExecutionPolicy())
    next_prepared = schedule_next_batch_decode_step(
        model,
        state,
        prepared,
        execution=CoreExecutionPolicy(),
    )
    token_ids = materialize_prepared_batch_step(
        prepared,
        execution=CoreExecutionPolicy(),
        next_prepared=next_prepared,
    )

    assert token_ids == (1, 2)
    assert tuple(int(v) for v in next_prepared.tokens.tolist()) == (2, 0)
    assert model.calls == [(2, 2), (2, 1), (2, 1)]
    assert state.cache[0].offset == 4


def test_filter_prepared_batch_step_and_extract_row_cache() -> None:
    model = _BatchModel([mx.zeros((2, 1, 2), dtype=mx.float32)])
    state = BatchCoreState.create(model)
    state.cache[0].state = (
        mx.zeros((2, 1, 3, 4), dtype=mx.float32),
        mx.zeros((2, 1, 3, 4), dtype=mx.float32),
    )
    prepared = prepare_batch_decode_step(
        mx.array([[0.0, 1.0], [1.0, 0.0]], dtype=mx.float32),
        execution=CoreExecutionPolicy(),
        prime_tokens=False,
    )

    filtered = filter_prepared_batch_step(prepared, [1])
    extracted = state.extract_row(1)
    state.filter_rows([1])

    assert tuple(int(v) for v in filtered.tokens.tolist()) == (0,)
    assert extracted[0].offset == 3
    assert state.cache[0].offset == 3
    assert state.cache[0].state[0].shape[0] == 1
