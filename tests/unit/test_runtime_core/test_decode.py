"""Tests for Layer 1 decode-step progression."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

import mlxs.runtime_core.decode as runtime_decode
from mlxs.runtime_core.contracts import CoreFinishSignal
from mlxs.runtime_core.decode import (
    decode_step,
    materialize_prepared_step,
    prepare_decode_step,
    prepare_next_logits,
    schedule_next_decode_step,
)
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


def test_prepare_decode_step_async_evals_selected_token(monkeypatch) -> None:
    async_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(runtime_decode.mx, "async_eval", lambda *args: async_calls.append(args))

    prepared = prepare_decode_step(
        mx.array([[0.0, 1.0, 0.0]]),
        execution=CoreExecutionPolicy(),
    )

    assert len(async_calls) == 1
    assert async_calls[0] == (prepared.token,)
    assert int(prepared.token.item()) == 1
    assert prepared.logits.shape == (1, 3)


def test_prepare_decode_step_can_skip_async_priming(monkeypatch) -> None:
    async_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(runtime_decode.mx, "async_eval", lambda *args: async_calls.append(args))

    prepare_decode_step(
        mx.array([[0.0, 1.0, 0.0]]),
        execution=CoreExecutionPolicy(),
        prime_token=False,
    )

    assert async_calls == []


def test_schedule_next_decode_step_uses_step_override(monkeypatch) -> None:
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

    async_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(runtime_decode.mx, "async_eval", lambda *args: async_calls.append(args))

    model = _ShouldNotRunModel([])
    state = CoreState.adopt([_FakeCache()], prompt_tokens=4)
    prepared = prepare_decode_step(
        mx.array([[0.0, 1.0, 0.0]]),
        execution=CoreExecutionPolicy(),
    )
    async_calls.clear()

    next_prepared = schedule_next_decode_step(
        model,
        state,
        prepared,
        execution=CoreExecutionPolicy(),
        step_fn=lambda input_ids: mx.array([[[0.0, 0.0, 1.0]]]),
    )

    assert async_calls == [(next_prepared.token,)]
    assert int(next_prepared.token.item()) == 2
    assert next_prepared.logits.shape == (1, 3)


def test_prepare_next_logits_uses_step_override_and_can_prime(monkeypatch) -> None:
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

    async_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(runtime_decode.mx, "async_eval", lambda *args: async_calls.append(args))

    model = _ShouldNotRunModel([])
    state = CoreState.adopt([_FakeCache()], prompt_tokens=4)
    prepared = prepare_decode_step(
        mx.array([[0.0, 1.0, 0.0]]),
        execution=CoreExecutionPolicy(),
    )
    async_calls.clear()

    next_logits = prepare_next_logits(
        model,
        state,
        prepared,
        execution=CoreExecutionPolicy(),
        step_fn=lambda input_ids: mx.array([[[0.0, 0.0, 1.0]]]),
        prime_logits=True,
    )

    assert async_calls == [(next_logits.logits,)]
    assert next_logits.logits.shape == (1, 3)


def test_materialize_prepared_step_groups_eval_with_next_prepared(monkeypatch) -> None:
    eval_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(runtime_decode.mx, "eval", lambda *args: eval_calls.append(args))

    state = CoreState.adopt([_FakeCache()], prompt_tokens=3)
    prepared = prepare_decode_step(
        mx.array([[0.0, 1.0, 0.0]]),
        execution=CoreExecutionPolicy(),
        prime_token=False,
    )
    next_prepared = prepare_decode_step(
        mx.array([[0.0, 0.0, 1.0]]),
        execution=CoreExecutionPolicy(),
        prime_token=False,
    )

    result = materialize_prepared_step(
        state,
        prepared,
        termination=CoreTerminationPolicy(max_tokens=5, eos_token_ids=()),
        execution=CoreExecutionPolicy(),
        next_prepared=next_prepared,
    )

    assert eval_calls == [(prepared.token, next_prepared.token)]
    assert result.token_id == 1
    assert result.finish is None
    assert result.generation_tokens == 1


def test_materialize_prepared_step_forces_eval_and_updates_state(monkeypatch) -> None:
    eval_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(runtime_decode.mx, "eval", lambda *args: eval_calls.append(args))

    state = CoreState.adopt([_FakeCache()], prompt_tokens=3)
    prepared = prepare_decode_step(
        mx.array([[0.0, 1.0, 0.0]]),
        execution=CoreExecutionPolicy(),
    )

    result = materialize_prepared_step(
        state,
        prepared,
        termination=CoreTerminationPolicy(max_tokens=5, eos_token_ids=()),
        execution=CoreExecutionPolicy(),
        force_eval=True,
    )

    assert eval_calls == [(prepared.token,)]
    assert result.token_id == 1
    assert result.finish is None
    assert result.generation_tokens == 1


def test_materialize_prepared_step_clears_cache_on_interval(monkeypatch) -> None:
    clear_calls: list[str] = []
    monkeypatch.setattr(runtime_decode.mx, "clear_cache", lambda: clear_calls.append("clear"))

    state = CoreState.adopt([_FakeCache()], prompt_tokens=2, generation_tokens=1)
    prepared = prepare_decode_step(
        mx.array([[1.0, 0.0]]),
        execution=CoreExecutionPolicy(),
    )

    result = materialize_prepared_step(
        state,
        prepared,
        termination=CoreTerminationPolicy(max_tokens=2, eos_token_ids=()),
        execution=CoreExecutionPolicy(clear_cache_interval=2),
    )

    assert result.finish == CoreFinishSignal.LENGTH
    assert clear_calls == ["clear"]
