"""Tests for Layer 2 single-request helpers."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

import mlxs.general_path.single_request as single_request
from mlxs._types import GenerateOptions
from mlxs.general_path.single_request import (
    _build_top_logprobs,
    _select_token_from_logits,
    generate_single_request,
)
from mlxs.runtime_core.contracts import CoreFinishSignal, CoreStepResult


class _BatchTokenizer:
    def __init__(self) -> None:
        self.batch_decode_calls = 0
        self.decode_calls = 0
        self.inner = self

    def batch_decode(self, token_ids: list[list[int]]) -> list[str]:
        self.batch_decode_calls += 1
        return [f"tok-{ids[0]}" for ids in token_ids]

    def decode(self, token_ids: list[int] | int) -> str:
        self.decode_calls += 1
        if isinstance(token_ids, int):
            return f"tok-{token_ids}"
        return f"tok-{token_ids[0]}"


class _SimpleTokenizer:
    def __init__(self) -> None:
        self.decode_calls = 0

    def decode(self, token_ids: list[int] | int) -> str:
        self.decode_calls += 1
        if isinstance(token_ids, int):
            return f"tok-{token_ids}"
        return f"tok-{token_ids[0]}"


def test_build_top_logprobs_uses_batch_decode_when_available() -> None:
    tokenizer = _BatchTokenizer()
    logprobs = mx.array([-3.0, -0.2, -1.0, -0.5])

    out = _build_top_logprobs(tokenizer, logprobs, top_n=3)

    assert [item.token_id for item in out] == [1, 3, 2]
    assert [item.token for item in out] == ["tok-1", "tok-3", "tok-2"]
    assert tokenizer.batch_decode_calls == 1
    assert tokenizer.decode_calls == 0


def test_build_top_logprobs_falls_back_to_decode() -> None:
    tokenizer = _SimpleTokenizer()
    logprobs = mx.array([-3.0, -0.2, -1.0, -0.5])

    out = _build_top_logprobs(tokenizer, logprobs, top_n=2)

    assert [item.token_id for item in out] == [1, 3]
    assert [item.token for item in out] == ["tok-1", "tok-3"]
    assert tokenizer.decode_calls == 2


def test_select_token_from_logits_normalizes_before_sampling() -> None:
    seen: list[mx.array] = []

    def sampler(logprobs: mx.array) -> mx.array:
        seen.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    logits = mx.array([[1.0, 2.0, 0.5]])
    out = _select_token_from_logits(sampler, logits)

    expected = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(out, expected)
    assert len(seen) == 1
    assert mx.allclose(seen[0], expected).item() is True


class _GenTokenizer:
    eos_token_id = None

    def __init__(self) -> None:
        self.decode_calls = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: list[int] | int) -> str:
        self.decode_calls += 1
        if isinstance(token_ids, int):
            return f"tok-{token_ids}"
        return f"tok-{token_ids[0]}"


class _GenCache:
    @property
    def state(self) -> None:
        return None


class _GenModel:
    def make_cache(self) -> list[Any]:
        return [_GenCache()]


def test_generate_single_request_short_logprobs_uses_prepared_step(monkeypatch) -> None:
    prepare_calls: list[tuple[int, ...]] = []
    schedule_calls: list[int] = []
    materialize_calls: list[int] = []
    prepared_queue = [
        single_request.SimpleNamespace(logits=mx.array([[0.0, 1.0, 0.0]]), token=mx.array([1])),
        single_request.SimpleNamespace(logits=mx.array([[0.0, 0.0, 1.0]]), token=mx.array([2])),
    ]
    results = iter(
        [
            CoreStepResult(token_id=1, finish=None, prompt_tokens=3, generation_tokens=1),
            CoreStepResult(
                token_id=2,
                finish=CoreFinishSignal.LENGTH,
                prompt_tokens=3,
                generation_tokens=2,
            ),
        ]
    )

    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])
    monkeypatch.setattr(
        single_request,
        "decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("decode_step should not run on prepared-step branch")
        ),
    )

    def fake_prepare(logits: mx.array, **kwargs: Any) -> Any:
        del kwargs
        prepare_calls.append(tuple(float(v) for v in logits.reshape(-1).tolist()))
        return prepared_queue[0]

    def fake_schedule(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        schedule_calls.append(1)
        return prepared_queue[1]

    def fake_materialize(*args: Any, **kwargs: Any) -> CoreStepResult:
        del args, kwargs
        materialize_calls.append(1)
        return next(results)

    monkeypatch.setattr(single_request, "prepare_decode_step", fake_prepare)
    monkeypatch.setattr(single_request, "schedule_next_decode_step", fake_schedule)
    monkeypatch.setattr(single_request, "materialize_prepared_step", fake_materialize)

    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            [10, 11, 12],
            GenerateOptions(max_tokens=2, temperature=0.0, logprobs=True),
        )
    )

    assert [event.token_id for event in events] == [1, 2]
    assert events[-1].finish_reason is not None
    assert prepare_calls == [(1.0, 0.0, 0.0)]
    assert len(schedule_calls) == 1
    assert len(materialize_calls) == 2


def test_generate_single_request_short_logprobs_passes_execution_stream(monkeypatch) -> None:
    seen_streams: list[object | None] = []
    prepared_queue = [
        single_request.SimpleNamespace(logits=mx.array([[0.0, 1.0, 0.0]]), token=mx.array([1])),
    ]

    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])

    def fake_prepare(logits: mx.array, **kwargs: Any) -> Any:
        del logits
        seen_streams.append(kwargs["execution"].stream)
        return prepared_queue[0]

    monkeypatch.setattr(single_request, "prepare_decode_step", fake_prepare)
    monkeypatch.setattr(
        single_request,
        "schedule_next_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no next step expected")),
    )
    monkeypatch.setattr(
        single_request,
        "materialize_prepared_step",
        lambda *args, **kwargs: CoreStepResult(
            token_id=1,
            finish=CoreFinishSignal.LENGTH,
            prompt_tokens=3,
            generation_tokens=1,
        ),
    )

    stream = object()
    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            [10, 11, 12],
            GenerateOptions(max_tokens=1, temperature=0.0, logprobs=True),
            execution_stream=stream,
        )
    )

    assert [event.token_id for event in events] == [1]
    assert seen_streams == [stream]


def test_generate_single_request_short_sampled_path_uses_prepared_step(monkeypatch) -> None:
    schedule_calls: list[int] = []
    results = iter(
        [
            CoreStepResult(token_id=1, finish=None, prompt_tokens=3, generation_tokens=1),
            CoreStepResult(
                token_id=2,
                finish=CoreFinishSignal.LENGTH,
                prompt_tokens=3,
                generation_tokens=2,
            ),
        ]
    )

    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])
    monkeypatch.setattr(
        single_request,
        "decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("decode_step should not run on short sampled prepared branch")
        ),
    )
    monkeypatch.setattr(
        single_request,
        "prepare_decode_step",
        lambda *args, **kwargs: single_request.SimpleNamespace(
            logits=mx.array([[0.0, 1.0, 0.0]]),
            token=mx.array([1]),
        ),
    )
    monkeypatch.setattr(
        single_request,
        "schedule_next_decode_step",
        lambda *args, **kwargs: schedule_calls.append(1)
        or single_request.SimpleNamespace(
            logits=mx.array([[0.0, 0.0, 1.0]]),
            token=mx.array([2]),
        ),
    )
    monkeypatch.setattr(
        single_request,
        "materialize_prepared_step",
        lambda *args, **kwargs: next(results),
    )

    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            [10, 11, 12],
            GenerateOptions(max_tokens=2, temperature=0.8, top_k=40, logprobs=False),
            execution_stream=object(),
        )
    )

    assert [event.token_id for event in events] == [1, 2]
    assert [event.logprobs for event in events] == [None, None]
    assert schedule_calls == [1]


def test_generate_single_request_short_sampled_path_requires_execution_stream(
    monkeypatch,
) -> None:
    decode_calls: list[object | None] = []
    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])
    monkeypatch.setattr(
        single_request,
        "prepare_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sampled prepared-step branch should not run without execution_stream")
        ),
    )

    def fake_decode(*args: Any, **kwargs: Any) -> tuple[CoreStepResult, None]:
        del args
        decode_calls.append(kwargs["execution"].stream)
        return (
            CoreStepResult(
                token_id=1,
                finish=CoreFinishSignal.LENGTH,
                prompt_tokens=3,
                generation_tokens=1,
            ),
            None,
        )

    monkeypatch.setattr(single_request, "decode_step", fake_decode)

    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            [10, 11, 12],
            GenerateOptions(max_tokens=1, temperature=0.8, top_k=40, logprobs=False),
        )
    )

    assert [event.token_id for event in events] == [1]
    assert decode_calls == [None]


def test_generate_single_request_short_logprobs_with_processors_stays_on_decode_step(
    monkeypatch,
) -> None:
    decode_calls: list[int] = []
    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(
        single_request,
        "make_logits_processors",
        lambda **kwargs: [lambda tokens, logits: logits],
    )
    monkeypatch.setattr(
        single_request,
        "prepare_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepared-step branch should not run when processors are active")
        ),
    )

    def fake_decode(*args: Any, **kwargs: Any) -> tuple[CoreStepResult, None]:
        del args, kwargs
        decode_calls.append(1)
        return (
            CoreStepResult(
                token_id=1,
                finish=CoreFinishSignal.LENGTH,
                prompt_tokens=3,
                generation_tokens=1,
            ),
            None,
        )

    monkeypatch.setattr(single_request, "decode_step", fake_decode)

    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            [10, 11, 12],
            GenerateOptions(
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                repetition_penalty=1.1,
            ),
        )
    )

    assert [event.token_id for event in events] == [1]
    assert decode_calls == [1]


def test_generate_single_request_short_processor_path_uses_prepare_next_logits(
    monkeypatch,
) -> None:
    prepare_streams: list[object | None] = []
    next_logits_calls: list[object | None] = []
    results = iter(
        [
            CoreStepResult(
                token_id=1,
                finish=None,
                prompt_tokens=3,
                generation_tokens=1,
            ),
            CoreStepResult(
                token_id=2,
                finish=CoreFinishSignal.LENGTH,
                prompt_tokens=3,
                generation_tokens=2,
            ),
        ]
    )

    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(
        single_request,
        "make_logits_processors",
        lambda **kwargs: [lambda tokens, logits: logits],
    )
    monkeypatch.setattr(
        single_request,
        "decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("decode_step should not run on processor prepared branch")
        ),
    )

    def fake_prepare(logits: mx.array, **kwargs: Any) -> Any:
        del logits
        prepare_streams.append(kwargs["execution"].stream)
        return single_request.SimpleNamespace(
            logits=mx.array([[0.0, 1.0, 0.0]]),
            token=mx.array([1]),
        )

    monkeypatch.setattr(single_request, "prepare_decode_step", fake_prepare)
    monkeypatch.setattr(
        single_request,
        "prepare_next_logits",
        lambda *args, **kwargs: next_logits_calls.append(kwargs["execution"].stream)
        or single_request.SimpleNamespace(logits=mx.array([[0.0, 0.0, 1.0]])),
    )
    monkeypatch.setattr(
        single_request,
        "materialize_prepared_step",
        lambda *args, **kwargs: next(results),
    )

    stream = object()
    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            [10, 11, 12],
            GenerateOptions(
                max_tokens=2,
                temperature=0.0,
                logprobs=False,
                repetition_penalty=1.1,
            ),
            execution_stream=stream,
        )
    )

    assert [event.token_id for event in events] == [1, 2]
    assert prepare_streams == [stream, stream]
    assert next_logits_calls == [stream]


def test_generate_single_request_long_logprobs_ignores_execution_stream(monkeypatch) -> None:
    seen_streams: list[object | None] = []
    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])
    monkeypatch.setattr(
        single_request,
        "prepare_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepared-step branch should not run on long prompts")
        ),
    )

    def fake_decode(*args: Any, **kwargs: Any) -> tuple[CoreStepResult, None]:
        del args
        seen_streams.append(kwargs["execution"].stream)
        return (
            CoreStepResult(
                token_id=1,
                finish=CoreFinishSignal.LENGTH,
                prompt_tokens=513,
                generation_tokens=1,
            ),
            None,
        )

    monkeypatch.setattr(single_request, "decode_step", fake_decode)

    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            list(range(single_request.SHORT_LOGPROBS_PREPARED_STEP_THRESHOLD + 1)),
            GenerateOptions(max_tokens=1, temperature=0.0, logprobs=True),
            execution_stream=object(),
        )
    )

    assert [event.token_id for event in events] == [1]
    assert seen_streams == [None]


def test_generate_single_request_long_top_logprobs_uses_prepared_step(monkeypatch) -> None:
    seen_streams: list[object | None] = []
    prepared_queue = [
        single_request.SimpleNamespace(logits=mx.array([[0.0, 1.0, 0.0]]), token=mx.array([1])),
    ]

    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])
    monkeypatch.setattr(
        single_request,
        "decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("decode_step should not run on long top_logprobs prepared branch")
        ),
    )

    def fake_prepare(logits: mx.array, **kwargs: Any) -> Any:
        del logits
        seen_streams.append(kwargs["execution"].stream)
        return prepared_queue[0]

    monkeypatch.setattr(single_request, "prepare_decode_step", fake_prepare)
    monkeypatch.setattr(
        single_request,
        "schedule_next_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no next step expected")),
    )
    monkeypatch.setattr(
        single_request,
        "materialize_prepared_step",
        lambda *args, **kwargs: CoreStepResult(
            token_id=1,
            finish=CoreFinishSignal.LENGTH,
            prompt_tokens=513,
            generation_tokens=1,
        ),
    )

    stream = object()
    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            list(range(single_request.SHORT_LOGPROBS_PREPARED_STEP_THRESHOLD + 1)),
            GenerateOptions(
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            ),
            execution_stream=stream,
        )
    )

    assert [event.token_id for event in events] == [1]
    assert seen_streams == [stream]


def test_generate_single_request_long_logprobs_stays_on_decode_step(monkeypatch) -> None:
    decode_calls: list[int] = []
    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])
    monkeypatch.setattr(
        single_request,
        "prepare_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepared-step branch should not run on long prompts")
        ),
    )

    def fake_decode(*args: Any, **kwargs: Any) -> tuple[CoreStepResult, None]:
        del args, kwargs
        decode_calls.append(1)
        return (
            CoreStepResult(
                token_id=1,
                finish=CoreFinishSignal.LENGTH,
                prompt_tokens=513,
                generation_tokens=1,
            ),
            None,
        )

    monkeypatch.setattr(single_request, "decode_step", fake_decode)

    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            list(range(single_request.SHORT_LOGPROBS_PREPARED_STEP_THRESHOLD + 1)),
            GenerateOptions(max_tokens=1, temperature=0.0, logprobs=True),
        )
    )

    assert [event.token_id for event in events] == [1]
    assert decode_calls == [1]


def test_generate_single_request_short_compile_decode_uses_compiled_step(monkeypatch) -> None:
    compiled_calls: list[int] = []
    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])
    monkeypatch.setattr(
        single_request,
        "make_compiled_step",
        lambda *args, **kwargs: compiled_calls.append(1)
        or (lambda input_ids: mx.array([[[0.0, 0.0, 1.0]]])),
    )
    monkeypatch.setattr(
        single_request,
        "prepare_decode_step",
        lambda *args, **kwargs: single_request.SimpleNamespace(
            logits=mx.array([[0.0, 1.0, 0.0]]),
            token=mx.array([1]),
        ),
    )
    monkeypatch.setattr(
        single_request,
        "schedule_next_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no next step expected")),
    )
    monkeypatch.setattr(
        single_request,
        "materialize_prepared_step",
        lambda *args, **kwargs: CoreStepResult(
            token_id=1,
            finish=CoreFinishSignal.LENGTH,
            prompt_tokens=3,
            generation_tokens=1,
        ),
    )

    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            [10, 11, 12],
            GenerateOptions(max_tokens=1, temperature=0.0, logprobs=False),
            compile_decode=True,
            execution_stream=object(),
        )
    )

    assert len(events) == 1
    assert compiled_calls == [1]


def test_generate_single_request_long_compile_decode_skips_compiled_step(monkeypatch) -> None:
    monkeypatch.setattr(
        single_request,
        "run_prefill",
        lambda *args, **kwargs: mx.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(single_request, "make_logits_processors", lambda **kwargs: [])
    monkeypatch.setattr(
        single_request,
        "make_compiled_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compiled step should not be built on long prompt")
        ),
    )
    decode_calls: list[int] = []

    def fake_decode(*args: Any, **kwargs: Any) -> tuple[CoreStepResult, None]:
        del args
        decode_calls.append(1)
        assert kwargs["step_fn"] is None
        return (
            CoreStepResult(
                token_id=1,
                finish=CoreFinishSignal.LENGTH,
                prompt_tokens=513,
                generation_tokens=1,
            ),
            None,
        )

    monkeypatch.setattr(single_request, "decode_step", fake_decode)

    events = list(
        generate_single_request(
            _GenModel(),
            _GenTokenizer(),
            list(range(single_request.SHORT_LOGPROBS_PREPARED_STEP_THRESHOLD + 1)),
            GenerateOptions(max_tokens=1, temperature=0.0, logprobs=False),
            compile_decode=True,
        )
    )

    assert [event.token_id for event in events] == [1]
    assert decode_calls == [1]
