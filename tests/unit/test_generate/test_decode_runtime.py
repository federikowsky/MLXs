"""Focused tests for the decode engine runtime."""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import pytest

from mlxs._types import FinishReason, GenerateOptions
from mlxs.generate import generate


class _RuntimeTokenizer:
    def __init__(self, mapping: dict[int, str], *, eos_token_id: int) -> None:
        self._mapping = mapping
        self._eos_token_id = eos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return self._mapping[token_ids]
        return "".join(self._mapping[token_id] for token_id in token_ids)


class _RuntimeCache:
    def __init__(self, label: str = "orig") -> None:
        self.calls = 0
        self.label = label

    @property
    def state(self) -> Any:
        return None


class _RuntimeModel:
    def __init__(self, predictions: list[int], *, vocab_size: int = 8) -> None:
        self._predictions = predictions
        self._vocab_size = vocab_size
        self.call_index = 0
        self.cache_markers: list[str] = []

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[_RuntimeCache] | None = None,
        mask: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        del mask, input_embeddings
        assert cache is not None
        self.cache_markers.append(cache[0].label)
        cache[0].calls += 1
        token = self._predictions[min(self.call_index, len(self._predictions) - 1)]
        self.call_index += 1

        base = -mx.arange(self._vocab_size, dtype=mx.float32)
        row = mx.where(
            mx.arange(self._vocab_size) == token,
            mx.array(5.0, dtype=mx.float32),
            base - 5.0,
        )
        return mx.broadcast_to(row, (input_ids.shape[0], input_ids.shape[1], self._vocab_size))

    def make_cache(self) -> list[_RuntimeCache]:
        return [_RuntimeCache()]


class _PenaltyModel:
    def __init__(self) -> None:
        self.call_index = 0

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[_RuntimeCache] | None = None,
        mask: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        del input_ids, mask, input_embeddings
        assert cache is not None
        cache[0].calls += 1
        idx = self.call_index
        self.call_index += 1

        row = mx.full((8,), -10.0, dtype=mx.float32)
        if idx in {0, 1}:
            row[7] = 6.0
            row[5] = 5.0
        else:
            row[0] = 6.0
            row[5] = 4.0
        return mx.broadcast_to(row, (1, 1, 8))

    def make_cache(self) -> list[_RuntimeCache]:
        return [_RuntimeCache()]


def test_final_token_stop_does_not_dispatch_next_step() -> None:
    model = _RuntimeModel([0, 7])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)
    final_cache: list[list[_RuntimeCache]] = []

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=3, temperature=0.0),
            final_cache_out=final_cache,
        )
    )

    assert [event.token_id for event in events] == [0]
    assert events[0].finish_reason is FinishReason.STOP
    assert model.call_index == 1
    assert final_cache[0][0].calls == 1


def test_stop_sequence_match_across_token_boundaries() -> None:
    model = _RuntimeModel([1, 2, 3])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 1: "he", 2: "llo", 3: "!"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(
                max_tokens=5,
                temperature=0.0,
                stop_sequences=("hello",),
            ),
        )
    )

    assert [event.text for event in events] == ["he", "llo"]
    assert events[-1].finish_reason is FinishReason.STOP
    assert model.call_index == 2


def test_compile_on_off_parity_for_tokens_finish_and_logprobs(monkeypatch: Any) -> None:
    def _fake_make_compiled_step_backend(
        model: _RuntimeModel,
        cache: list[_RuntimeCache],
        recipe: Any,
    ) -> Any:
        def _step(input_ids: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
            logits = model(input_ids, cache=cache)[:, -1, :]
            token = mx.argmax(logits, axis=-1)
            lse = mx.logsumexp(logits, keepdims=True)
            token_logprob = mx.take_along_axis(logits, token[:, None], axis=-1).reshape(-1)
            token_logprob = token_logprob - lse.reshape(-1)
            row = logits[0]
            top_ids = mx.argpartition(row, kth=-recipe.top_logprobs)[-recipe.top_logprobs :]
            top_ids = top_ids[mx.argsort(row[top_ids])[::-1]]
            top_logprobs = row[top_ids] - lse[0, 0]
            return token, token_logprob, top_ids, top_logprobs

        return _step

    monkeypatch.setattr(
        "mlxs.generate.compile.make_compiled_step_backend",
        _fake_make_compiled_step_backend,
    )

    options = GenerateOptions(
        max_tokens=3,
        temperature=0.0,
        logprobs=True,
        top_logprobs=3,
    )
    token_map = {idx: f"tok{idx}" for idx in range(8)}
    token_map[0] = "<eos>"

    def _run(*, compile_decode: bool) -> list[Any]:
        model = _RuntimeModel([7, 5, 0])
        tokenizer = _RuntimeTokenizer(token_map, eos_token_id=0)
        return list(
            generate(
                model,
                tokenizer,
                [1],
                options,
                compile_decode=compile_decode,
            )
        )

    events_off = _run(compile_decode=False)
    events_on = _run(compile_decode=True)

    assert [event.token_id for event in events_off] == [event.token_id for event in events_on]
    assert [event.finish_reason for event in events_off] == [
        event.finish_reason for event in events_on
    ]
    assert [event.prompt_tokens for event in events_off] == [
        event.prompt_tokens for event in events_on
    ]
    assert [event.generation_tokens for event in events_off] == [
        event.generation_tokens for event in events_on
    ]

    for off, on in zip(events_off, events_on, strict=True):
        assert off.logprobs is not None
        assert on.logprobs is not None
        assert off.logprobs.token_logprob == pytest.approx(on.logprobs.token_logprob)
        assert [entry.token_id for entry in off.logprobs.top_logprobs] == [
            entry.token_id for entry in on.logprobs.top_logprobs
        ]
        assert [entry.token for entry in off.logprobs.top_logprobs] == [
            entry.token for entry in on.logprobs.top_logprobs
        ]
        assert [entry.logprob for entry in off.logprobs.top_logprobs] == pytest.approx(
            [entry.logprob for entry in on.logprobs.top_logprobs]
        )


def test_repetition_penalty_parity_with_device_history() -> None:
    tokenizer = _RuntimeTokenizer({0: "<eos>", 5: "five", 7: "seven"}, eos_token_id=0)
    options = GenerateOptions(
        max_tokens=3,
        temperature=0.0,
        repetition_penalty=2.0,
    )

    def _run(*, compile_decode: bool) -> list[Any]:
        model = _PenaltyModel()
        return list(
            generate(
                model,
                tokenizer,
                [1],
                options,
                compile_decode=compile_decode,
            )
        )

    events_off = _run(compile_decode=False)
    events_on = _run(compile_decode=True)

    assert [event.token_id for event in events_off] == [7, 5, 0]
    assert [event.token_id for event in events_off] == [event.token_id for event in events_on]
    assert [event.finish_reason for event in events_off] == [
        event.finish_reason for event in events_on
    ]


def test_cache_replacement_occurs_before_next_dispatch(monkeypatch: Any) -> None:
    def _fake_convert_to_quantized(
        cache: list[_RuntimeCache],
        *,
        kv_bits: int,
        kv_group_size: int,
    ) -> list[_RuntimeCache]:
        del kv_bits, kv_group_size
        return [_RuntimeCache("quantized") for _ in cache]

    monkeypatch.setattr("mlxs.cache.convert_to_quantized", _fake_convert_to_quantized)

    model = _RuntimeModel([7, 5, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 5: "five", 7: "seven"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=3, temperature=0.0),
            quantized_kv_start=1,
            kv_bits=8,
        )
    )

    assert [event.token_id for event in events] == [7, 5, 0]
    assert model.cache_markers == ["orig", "orig", "quantized"]


def test_compile_rebind_failure_falls_back_to_uncompiled(monkeypatch: Any) -> None:
    from mlxs.generate.compile import make_compiled_step_backend as real_builder

    build_count = 0

    def _counting_builder(model: _RuntimeModel, cache: list[_RuntimeCache], recipe: Any) -> Any:
        nonlocal build_count
        build_count += 1
        if build_count == 1:
            return real_builder(model, cache, recipe)
        raise RuntimeError("rebind exploded")

    def _fake_convert_to_quantized(
        cache: list[_RuntimeCache],
        *,
        kv_bits: int,
        kv_group_size: int,
    ) -> list[_RuntimeCache]:
        del kv_bits, kv_group_size
        return [_RuntimeCache("quantized") for _ in cache]

    monkeypatch.setattr("mlxs.generate.compile.make_compiled_step_backend", _counting_builder)
    monkeypatch.setattr("mlxs.cache.convert_to_quantized", _fake_convert_to_quantized)

    model = _RuntimeModel([7, 5, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 5: "five", 7: "seven"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=3, temperature=0.0),
            compile_decode=True,
            quantized_kv_start=1,
            kv_bits=8,
        )
    )

    assert build_count == 2
    assert [event.token_id for event in events] == [7, 5, 0]
    assert model.cache_markers == ["orig", "orig", "quantized"]


def test_async_boundary_keeps_seed_sync_and_enqueues_steady_state(monkeypatch: Any) -> None:
    calls: list[tuple[str, int]] = []
    real_async_eval = cast(Any, mx.async_eval)
    real_eval = mx.eval

    def _wrapped_async_eval(*args: Any) -> Any:
        calls.append(("async", len(args)))
        return real_async_eval(*args)

    def _wrapped_eval(*args: Any) -> Any:
        calls.append(("sync", len(args)))
        return real_eval(*args)

    monkeypatch.setattr("mlxs.generate.runtime.mx.async_eval", _wrapped_async_eval)
    monkeypatch.setattr("mlxs.generate.runtime.mx.eval", _wrapped_eval)

    model = _RuntimeModel([7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=2, temperature=0.0),
        )
    )

    assert [event.token_id for event in events] == [7, 0]
    assert calls == [("sync", 1), ("async", 1), ("sync", 1)]


def test_sync_boundary_can_be_forced_via_env(monkeypatch: Any) -> None:
    calls: list[tuple[str, int]] = []
    real_async_eval = cast(Any, mx.async_eval)
    real_eval = mx.eval

    def _wrapped_async_eval(*args: Any) -> Any:
        calls.append(("async", len(args)))
        return real_async_eval(*args)

    def _wrapped_eval(*args: Any) -> Any:
        calls.append(("sync", len(args)))
        return real_eval(*args)

    monkeypatch.setenv("MLXS_DECODE_ASYNC_EVAL", "0")
    monkeypatch.setattr("mlxs.generate.runtime.mx.async_eval", _wrapped_async_eval)
    monkeypatch.setattr("mlxs.generate.runtime.mx.eval", _wrapped_eval)

    model = _RuntimeModel([7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=2, temperature=0.0),
        )
    )

    assert [event.token_id for event in events] == [7, 0]
    assert calls[-2:] == [("sync", 1), ("sync", 1)]


def test_heavy_uncompiled_boundary_defaults_to_sync(monkeypatch: Any) -> None:
    calls: list[tuple[str, int]] = []
    real_async_eval = cast(Any, mx.async_eval)
    real_eval = mx.eval

    def _wrapped_async_eval(*args: Any) -> Any:
        calls.append(("async", len(args)))
        return real_async_eval(*args)

    def _wrapped_eval(*args: Any) -> Any:
        calls.append(("sync", len(args)))
        return real_eval(*args)

    monkeypatch.delenv("MLXS_DECODE_ASYNC_EVAL", raising=False)
    monkeypatch.setattr("mlxs.generate.runtime.mx.async_eval", _wrapped_async_eval)
    monkeypatch.setattr("mlxs.generate.runtime.mx.eval", _wrapped_eval)

    model = _RuntimeModel([6, 7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 6: "six", 7: "seven"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1] * 2048,
            GenerateOptions(max_tokens=2, temperature=0.0),
        )
    )

    assert [event.token_id for event in events] == [7, 0]
    assert calls[-2:] == [("sync", 1), ("sync", 1)]


def test_heavy_compiled_boundary_stays_async(monkeypatch: Any) -> None:
    calls: list[tuple[str, int]] = []
    real_async_eval = cast(Any, mx.async_eval)
    real_eval = mx.eval

    def _wrapped_async_eval(*args: Any) -> Any:
        calls.append(("async", len(args)))
        return real_async_eval(*args)

    def _wrapped_eval(*args: Any) -> Any:
        calls.append(("sync", len(args)))
        return real_eval(*args)

    def _fake_make_compiled_step_backend(
        model: _RuntimeModel,
        cache: list[_RuntimeCache],
        recipe: Any,
    ) -> Any:
        del recipe

        def _step(input_ids: mx.array) -> mx.array:
            return mx.argmax(model(input_ids, cache=cache)[:, -1, :], axis=-1)

        return _step

    monkeypatch.delenv("MLXS_DECODE_ASYNC_EVAL", raising=False)
    monkeypatch.setattr("mlxs.generate.runtime.mx.async_eval", _wrapped_async_eval)
    monkeypatch.setattr("mlxs.generate.runtime.mx.eval", _wrapped_eval)
    monkeypatch.setattr(
        "mlxs.generate.compile.make_compiled_step_backend",
        _fake_make_compiled_step_backend,
    )

    model = _RuntimeModel([6, 7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 6: "six", 7: "seven"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1] * 2048,
            GenerateOptions(max_tokens=2, temperature=0.0),
            compile_decode=True,
        )
    )

    assert [event.token_id for event in events] == [7, 0]
    assert calls[-3:] == [("sync", 1), ("async", 1), ("sync", 1)]
