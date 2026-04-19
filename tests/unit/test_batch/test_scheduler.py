from __future__ import annotations

import mlx.core as mx

from mlxs._types import FinishReason, GenerateOptions
from mlxs.batch.scheduler import BatchScheduler
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache


class _DummyTokenizer:
    eos_token_id = 999

    def encode(self, prompt: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_id: int) -> str:
        return f"t{token_id}"


class _ProbeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.vocab_size = 4

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(2)]

    def __call__(self, input_ids, *, cache=None):
        shape = tuple(int(x) for x in input_ids.shape)
        self.calls.append(shape)
        if cache is not None:
            batch, seq_len = shape
            for layer in cache:
                keys = mx.zeros((batch, 1, seq_len, 4), dtype=mx.float32)
                values = mx.zeros((batch, 1, seq_len, 4), dtype=mx.float32)
                layer.update_and_fetch(keys, values)
        token_logits = mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        return mx.broadcast_to(token_logits, (shape[0], shape[1], self.vocab_size))


class _ScriptedBatchModel(_ProbeModel):
    def __init__(self, outputs: list[mx.array]) -> None:
        super().__init__()
        self._outputs = outputs

    def __call__(self, input_ids, *, cache=None):
        shape = tuple(int(x) for x in input_ids.shape)
        self.calls.append(shape)
        if cache is not None:
            batch, seq_len = shape
            for layer in cache:
                keys = mx.zeros((batch, 1, seq_len, 4), dtype=mx.float32)
                values = mx.zeros((batch, 1, seq_len, 4), dtype=mx.float32)
                layer.update_and_fetch(keys, values)
        return self._outputs.pop(0)


class _RotatingProbeModel(_ProbeModel):
    def make_cache(self) -> list[RotatingKVCache]:
        return [RotatingKVCache(max_size=16)]


class _EOSTokenizer(_DummyTokenizer):
    eos_token_id = 3


def test_scheduler_batches_aligned_decode_sequences() -> None:
    model = _ProbeModel()
    tokenizer = _DummyTokenizer()
    options = GenerateOptions(
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=0,
        stop_sequences=(),
        extra_eos_token_ids=(),
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        stream=True,
    )
    scheduler = BatchScheduler(
        prefill_batch_size=2,
        completion_batch_size=2,
        prefill_step_size=2048,
    )
    scheduler.add("a", model, tokenizer, [10, 11, 12], options)
    scheduler.add("b", model, tokenizer, [10, 11, 12], options)

    first = scheduler.step()

    assert set(first) == {"a", "b"}
    assert scheduler._fast_batch is not None
    assert not scheduler._active

    while scheduler.active_count or scheduler.pending_count:
        scheduler.step()
        list(scheduler.drain())

    assert any(shape[0] == 2 and shape[1] > 1 for shape in model.calls)
    assert (2, 1) in model.calls
    assert max(shape[0] for shape in model.calls) == 2


def test_scheduler_adopts_imported_cache_and_preserves_full_prompt_count() -> None:
    model = _ProbeModel()
    tokenizer = _DummyTokenizer()
    options = GenerateOptions(max_tokens=2, temperature=0.0)
    scheduler = BatchScheduler(
        prefill_batch_size=1,
        completion_batch_size=1,
        prefill_step_size=2048,
    )
    imported_cache = model.make_cache()
    for layer in imported_cache:
        keys = mx.zeros((1, 1, 3, 4), dtype=mx.float32)
        values = mx.zeros((1, 1, 3, 4), dtype=mx.float32)
        layer.update_and_fetch(keys, values)

    scheduler.add(
        "a",
        model,
        tokenizer,
        [10, 11],
        options,
        cache_state=imported_cache,
        prompt_token_count=5,
    )

    while scheduler.active_count or scheduler.pending_count:
        scheduler.step()

    drained = list(scheduler.drain())
    assert len(drained) == 1
    request_id, events, final_cache = drained[0]
    assert request_id == "a"
    assert final_cache is imported_cache
    assert events[0].prompt_tokens == 5
    assert final_cache[0].offset > 3


def test_scheduler_falls_back_for_rotating_cache_family() -> None:
    model = _RotatingProbeModel()
    tokenizer = _DummyTokenizer()
    options = GenerateOptions(max_tokens=3, temperature=0.0)
    scheduler = BatchScheduler(prefill_batch_size=2, completion_batch_size=2, prefill_step_size=2048)
    scheduler.add("a", model, tokenizer, [10, 11, 12], options)
    scheduler.add("b", model, tokenizer, [10, 11, 12], options)

    results = scheduler.step()

    assert scheduler._fast_batch is None
    assert set(results) == {"a", "b"}
    assert scheduler.active_count == 2


def test_scheduler_cancels_request_inside_fast_batch() -> None:
    model = _ProbeModel()
    tokenizer = _DummyTokenizer()
    options = GenerateOptions(max_tokens=4, temperature=0.0)
    scheduler = BatchScheduler(prefill_batch_size=2, completion_batch_size=2, prefill_step_size=2048)
    scheduler.add("a", model, tokenizer, [10, 11, 12], options)
    scheduler.add("b", model, tokenizer, [10, 11, 12], options)

    scheduler.step()
    assert scheduler._fast_batch is not None

    scheduler.remove("a")

    drained = list(scheduler.drain())
    assert len(drained) == 1
    request_id, events, final_cache = drained[0]
    assert request_id == "a"
    assert events[-1].finish_reason is FinishReason.CANCELLED
    assert final_cache is not None
    assert scheduler._fast_batch is not None
    assert len(scheduler._fast_batch) == 1


def test_scheduler_extracts_finished_row_from_fast_batch() -> None:
    model = _ScriptedBatchModel(
        [
            mx.zeros((2, 2, 4), dtype=mx.float32),
            mx.array([[[0.0, 1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]], dtype=mx.float32),
            mx.array([[[0.0, 0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 0.0]]], dtype=mx.float32),
            mx.array([[[0.0, 1.0, 0.0, 0.0]]], dtype=mx.float32),
        ]
    )
    tokenizer = _EOSTokenizer()
    options = GenerateOptions(max_tokens=4, temperature=0.0)
    scheduler = BatchScheduler(prefill_batch_size=2, completion_batch_size=2, prefill_step_size=2048)
    scheduler.add("a", model, tokenizer, [10, 11, 12], options)
    scheduler.add("b", model, tokenizer, [10, 11, 12], options)

    first = scheduler.step()
    assert set(first) == {"a", "b"}
    assert scheduler._fast_batch is not None

    second = scheduler.step()

    assert set(second) == {"a", "b"}
    drained = list(scheduler.drain())
    assert len(drained) == 1
    request_id, events, final_cache = drained[0]
    assert request_id == "a"
    assert [event.finish_reason for event in events] == [None, FinishReason.STOP]
    assert final_cache is not None
    assert final_cache[0].offset == 5
    assert scheduler._fast_batch is not None
    assert len(scheduler._fast_batch) == 1
