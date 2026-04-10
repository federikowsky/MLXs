"""Tests for the canonical Layer 1 greedy path."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlxs._types import TokenEvent
from mlxs.runtime_core.contracts import CoreFinishSignal, CoreStepResult
from mlxs.runtime_core.greedy import run_greedy
from mlxs.runtime_core.policy import CoreExecutionPolicy, CoreTerminationPolicy


class _FakeCache:
    @property
    def state(self) -> Any:
        return None


class _GreedyModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def make_cache(self) -> list[Any]:
        return [_FakeCache()]

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Any = None,
        input_embeddings: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        del cache, input_embeddings, mask
        flat = tuple(int(v) for v in input_ids.reshape(-1).tolist())
        self.calls.append(flat)
        batch, length = input_ids.shape
        if len(self.calls) == 1:
            return mx.zeros((batch, length, 3))
        if len(self.calls) == 2:
            return mx.array([[[0.0, 1.0, 0.0]]])
        return mx.array([[[0.0, 0.0, 1.0]]])


def test_run_greedy_emits_core_results_only() -> None:
    model = _GreedyModel()
    results = list(
        run_greedy(
            model,
            [11, 12],
            termination=CoreTerminationPolicy(max_tokens=4, eos_token_ids=(2,)),
            execution=CoreExecutionPolicy(),
        )
    )
    assert [type(result) for result in results] == [CoreStepResult, CoreStepResult]
    assert all(not isinstance(result, TokenEvent) for result in results)
    assert results[0].token_id == 1
    assert results[0].finish is None
    assert results[1].token_id == 2
    assert results[1].finish == CoreFinishSignal.STOP
    assert results[1].prompt_tokens == 2
    assert results[1].generation_tokens == 2
