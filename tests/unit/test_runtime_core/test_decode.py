"""Tests for Layer 1 decode-step progression."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

import mlxs.runtime_core.decode as runtime_decode
from mlxs.runtime_core.contracts import CoreFinishSignal
from mlxs.runtime_core.decode import decode_step
from mlxs.runtime_core.policy import CoreExecutionPolicy, CoreTerminationPolicy
from mlxs.runtime_core.state import CoreState


class _FakeCache:
    pass


class _QueuedModel:
    def __init__(self, outputs: list[mx.array]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[int, ...]] = []

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Any = None,
        input_embeddings: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        del cache, input_embeddings, mask
        self.calls.append(tuple(int(v) for v in input_ids.reshape(-1).tolist()))
        return self.outputs.pop(0)


def test_decode_step_advances_and_calls_model() -> None:
    model = _QueuedModel([mx.array([[[0.1, 0.2, 0.9]]])])
    state = CoreState.adopt([_FakeCache()], prompt_tokens=3)
    result, next_logits = decode_step(
        model,
        state,
        mx.array([[0.1, 0.9, 0.0]]),
        termination=CoreTerminationPolicy(max_tokens=5, eos_token_ids=(2,)),
        execution=CoreExecutionPolicy(),
    )
    assert result.token_id == 1
    assert result.finish is None
    assert result.prompt_tokens == 3
    assert result.generation_tokens == 1
    assert next_logits is not None
    assert next_logits.shape == (1, 3)
    assert model.calls == [(1,)]


def test_decode_step_stops_on_eos_without_step_call() -> None:
    model = _QueuedModel([])
    state = CoreState.adopt([_FakeCache()], prompt_tokens=2)
    result, next_logits = decode_step(
        model,
        state,
        mx.array([[0.1, 0.2, 0.9]]),
        termination=CoreTerminationPolicy(max_tokens=5, eos_token_ids=(2,)),
        execution=CoreExecutionPolicy(),
    )
    assert result.token_id == 2
    assert result.finish == CoreFinishSignal.STOP
    assert next_logits is None
    assert model.calls == []


def test_decode_step_stops_on_length_without_step_call() -> None:
    model = _QueuedModel([])
    state = CoreState.adopt([_FakeCache()], prompt_tokens=2, generation_tokens=1)
    result, next_logits = decode_step(
        model,
        state,
        mx.array([[0.9, 0.1, 0.0]]),
        termination=CoreTerminationPolicy(max_tokens=2, eos_token_ids=()),
        execution=CoreExecutionPolicy(),
    )
    assert result.token_id == 0
    assert result.finish == CoreFinishSignal.LENGTH
    assert result.generation_tokens == 2
    assert next_logits is None
    assert model.calls == []


def test_decode_step_clears_cache_on_interval(monkeypatch) -> None:
    clear_calls: list[str] = []
    monkeypatch.setattr(runtime_decode.mx, "clear_cache", lambda: clear_calls.append("clear"))

    model = _QueuedModel([])
    state = CoreState.adopt([_FakeCache()], prompt_tokens=2, generation_tokens=1)
    result, next_logits = decode_step(
        model,
        state,
        mx.array([[0.9, 0.1]]),
        termination=CoreTerminationPolicy(max_tokens=2, eos_token_ids=()),
        execution=CoreExecutionPolicy(clear_cache_interval=2),
    )
    assert result.finish == CoreFinishSignal.LENGTH
    assert next_logits is None
    assert clear_calls == ["clear"]


def test_decode_step_uses_step_override() -> None:
    class _ShouldNotRunModel(_QueuedModel):
        def __call__(
            self,
            input_ids: mx.array,
            *,
            cache: Any = None,
            input_embeddings: mx.array | None = None,
            mask: mx.array | None = None,
        ) -> mx.array:
            del input_ids, cache, input_embeddings, mask
            raise AssertionError("model step should not be used when step_fn is provided")

    model = _ShouldNotRunModel([])
    state = CoreState.adopt([_FakeCache()], prompt_tokens=4)
    result, next_logits = decode_step(
        model,
        state,
        mx.array([[0.0, 1.0, 0.0]]),
        termination=CoreTerminationPolicy(max_tokens=5, eos_token_ids=()),
        execution=CoreExecutionPolicy(),
        step_fn=lambda input_ids: mx.array([[[0.0, 0.0, 1.0]]]),
    )
    assert result.token_id == 1
    assert result.finish is None
    assert next_logits is not None
    assert next_logits.shape == (1, 3)
