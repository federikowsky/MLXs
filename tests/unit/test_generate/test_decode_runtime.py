"""Focused tests for the staged single-request decode runtime."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import pytest

from mlxs._types import FinishReason, GenerateOptions
from mlxs.generate import compile as compile_mod
from mlxs.generate import decode as decode_mod
from mlxs.generate import generate
from mlxs.protocols.generate import TokenizerProtocol


@dataclass(slots=True)
class _FakeCache:
    kind: str = "full"
    offset: int = 0

    @property
    def state(self) -> None:
        return None


class _FakeTokenizer:
    def __init__(self, token_text: dict[int, str], *, eos_token_id: int | None = 0) -> None:
        self._token_text = token_text
        self._eos_token_id = eos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        return max(self._token_text) + 1

    def encode(self, text: str) -> list[int]:
        raise AssertionError("Tests pass token ids directly.")

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return self._token_text[token_ids]
        return "".join(self._token_text[token_id] for token_id in token_ids)


class _TableModel:
    def __init__(self, row_for_token: Callable[[int], Sequence[float]]) -> None:
        self._row_for_token = row_for_token
        self.cache_kinds_seen: list[str] = []
        self.last_created_cache: list[_FakeCache] | None = None

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[_FakeCache] | None = None,
        mask: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        del mask, input_embeddings
        if cache is not None:
            self.cache_kinds_seen.append(cache[0].kind)
            for layer in cache:
                layer.offset += input_ids.shape[1]

        rows = [
            mx.array(list(self._row_for_token(int(token.item()))), dtype=mx.float32)
            for token in input_ids[0]
        ]
        return mx.stack(rows, axis=0)[None, :, :]

    def make_cache(self) -> list[_FakeCache]:
        self.last_created_cache = [_FakeCache()]
        return self.last_created_cache


def _row_with_winner(
    vocab_size: int,
    winner: int,
    *,
    winner_logit: float = 10.0,
    default_logit: float = -100.0,
) -> tuple[float, ...]:
    row = [default_logit] * vocab_size
    row[winner] = winner_logit
    return tuple(row)


def _transition_row_fn(
    transitions: dict[int, int],
    *,
    vocab_size: int,
    default_token: int = 0,
) -> Callable[[int], Sequence[float]]:
    return lambda token_id: _row_with_winner(
        vocab_size,
        transitions.get(token_id, default_token),
    )


def _event_summary(events: Sequence[Any]) -> list[tuple[int, str, FinishReason | None]]:
    return [(event.token_id, event.text, event.finish_reason) for event in events]


def test_generate_baseline_sequence_parity() -> None:
    tokenizer = _FakeTokenizer({0: "<eos>", 1: "A", 2: "B", 3: "C", 9: "P"})
    model = _TableModel(
        _transition_row_fn({9: 1, 1: 2, 2: 3, 3: 0}, vocab_size=10),
    )

    events = list(
        generate(
            model,
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=8, temperature=0),
        )
    )

    assert _event_summary(events) == [
        (1, "A", None),
        (2, "B", None),
        (3, "C", None),
        (0, "<eos>", FinishReason.STOP),
    ]


def test_generate_compile_on_off_have_same_output(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer({0: "<eos>", 1: "A", 2: "B", 3: "C", 9: "P"})
    transitions = _transition_row_fn({9: 1, 1: 2, 2: 3, 3: 0}, vocab_size=10)
    compile_builds: list[str] = []

    def fake_make_compiled_step(
        model: Any,
        cache: list[_FakeCache],
    ) -> Callable[[mx.array], mx.array]:
        compile_builds.append(cache[0].kind)
        return lambda input_ids: model(input_ids, cache=cache)

    monkeypatch.setattr(compile_mod, "make_compiled_step", fake_make_compiled_step)

    baseline_events = list(
        generate(
            _TableModel(transitions),
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=8, temperature=0),
            compile_decode=False,
        )
    )
    compiled_events = list(
        generate(
            _TableModel(transitions),
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=8, temperature=0),
            compile_decode=True,
        )
    )

    assert _event_summary(compiled_events) == _event_summary(baseline_events)
    assert compile_builds == ["full"]


def test_repetition_penalty_uses_bounded_recent_history() -> None:
    vocab_size = 24
    token_text = {0: "<eos>", **{token_id: str(token_id) for token_id in range(1, vocab_size)}}

    def row_for_token(_: int) -> Sequence[float]:
        row = [-100.0] * vocab_size
        for token_id in range(1, 22):
            row[token_id] = 10.0
        return row

    events = list(
        generate(
            _TableModel(row_for_token),
            cast(TokenizerProtocol, _FakeTokenizer(token_text)),
            [23],
            GenerateOptions(max_tokens=22, temperature=0, repetition_penalty=2.0),
        )
    )

    assert [event.token_id for event in events] == [*range(1, 22), 1]
    assert events[-1].finish_reason == FinishReason.LENGTH


def test_generate_stops_on_extra_eos_token() -> None:
    tokenizer = _FakeTokenizer({0: "<eos>", 7: "<stop>", 9: "P"})
    model = _TableModel(_transition_row_fn({9: 7}, vocab_size=10))

    events = list(
        generate(
            model,
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=5, temperature=0, extra_eos_token_ids=(7,)),
        )
    )

    assert _event_summary(events) == [(7, "<stop>", FinishReason.STOP)]


def test_generate_stops_on_max_tokens() -> None:
    tokenizer = _FakeTokenizer({0: "<eos>", 1: "A", 9: "P"})
    model = _TableModel(_transition_row_fn({9: 1, 1: 1}, vocab_size=10))

    events = list(
        generate(
            model,
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=3, temperature=0),
        )
    )

    assert _event_summary(events) == [
        (1, "A", None),
        (1, "A", None),
        (1, "A", FinishReason.LENGTH),
    ]


def test_generate_stops_on_stop_sequence() -> None:
    tokenizer = _FakeTokenizer({0: "<eos>", 1: "S", 2: "T", 3: "O", 4: "P", 9: "P"})
    model = _TableModel(_transition_row_fn({9: 1, 1: 2, 2: 3, 3: 4, 4: 0}, vocab_size=10))

    events = list(
        generate(
            model,
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=8, temperature=0, stop_sequences=("STOP",)),
        )
    )

    assert _event_summary(events) == [
        (1, "S", None),
        (2, "T", None),
        (3, "O", None),
        (4, "P", FinishReason.STOP),
    ]


def test_generate_emits_logprobs_and_top_logprobs() -> None:
    tokenizer = _FakeTokenizer({0: "<eos>", 1: "A", 2: "B", 3: "C", 4: "D", 9: "P"})

    def row_for_token(token_id: int) -> Sequence[float]:
        if token_id == 9:
            return (-100.0, 5.0, 4.0, 3.0, 0.0, -100.0)
        return _row_with_winner(6, 0)

    events = list(
        generate(
            _TableModel(row_for_token),
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=1, temperature=0, logprobs=True, top_logprobs=3),
        )
    )

    event = events[0]
    assert event.logprobs is not None
    assert event.token_id == 1
    assert [entry.token_id for entry in event.logprobs.top_logprobs] == [1, 2, 3]
    assert [entry.token for entry in event.logprobs.top_logprobs] == ["A", "B", "C"]

    expected = 5.0 - math.log(math.exp(5.0) + math.exp(4.0) + math.exp(3.0) + math.exp(0.0))
    assert event.logprobs.token_logprob == pytest.approx(expected)


def test_quantized_kv_start_rebuilds_forward_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer({0: "<eos>", 1: "A", 2: "B", 3: "C", 9: "P"})
    model = _TableModel(_transition_row_fn({9: 1, 1: 2, 2: 3, 3: 0}, vocab_size=10))
    compile_builds: list[str] = []

    def fake_make_compiled_step(
        model: Any,
        cache: list[_FakeCache],
    ) -> Callable[[mx.array], mx.array]:
        compile_builds.append(cache[0].kind)
        return lambda input_ids: model(input_ids, cache=cache)

    def fake_convert_to_quantized(
        cache: list[_FakeCache],
        *,
        kv_bits: int = 8,
        kv_group_size: int = 64,
    ) -> list[_FakeCache]:
        assert kv_bits == 4
        assert kv_group_size == 16
        return [_FakeCache(kind="quantized", offset=cache[0].offset)]

    monkeypatch.setattr(compile_mod, "make_compiled_step", fake_make_compiled_step)
    monkeypatch.setattr(decode_mod, "convert_to_quantized", fake_convert_to_quantized)

    events = list(
        generate(
            model,
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=8, temperature=0),
            compile_decode=True,
            quantized_kv_start=1,
            kv_bits=4,
            kv_group_size=16,
        )
    )

    assert _event_summary(events) == [
        (1, "A", None),
        (2, "B", None),
        (3, "C", None),
        (0, "<eos>", FinishReason.STOP),
    ]
    assert compile_builds == ["full", "quantized"]
    assert "quantized" in model.cache_kinds_seen


def test_clear_cache_interval_preserves_current_early_clear_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer({0: "<eos>", 1: "A", 9: "P"})
    model = _TableModel(_transition_row_fn({9: 1, 1: 1}, vocab_size=10))
    clear_offsets: list[int] = []

    def fake_clear_cache() -> None:
        assert model.last_created_cache is not None
        clear_offsets.append(model.last_created_cache[0].offset)

    cache = model.make_cache()
    monkeypatch.setattr("mlxs.generate.decode.mx.clear_cache", fake_clear_cache)

    events = list(
        generate(
            model,
            cast(TokenizerProtocol, tokenizer),
            [9],
            GenerateOptions(max_tokens=5, temperature=0),
            cache=cast(Any, cache),
            clear_cache_interval=2,
        )
    )

    assert [event.token_id for event in events] == [1, 1, 1, 1, 1]
    assert events[-1].finish_reason == FinishReason.LENGTH
    assert clear_offsets == [1, 3]
