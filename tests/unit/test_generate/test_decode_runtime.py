"""Focused tests for Decode Engine V4 Milestone 1."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

import mlx.core as mx
import pytest

import mlxs.generate.core as core_mod
from mlxs._types import FinishReason, GenerateOptions
from mlxs.cache.kv import KVCache
from mlxs.cache.quantized import QuantizedKVCache
from mlxs.generate import generate
from mlxs.generate.recipe import CompileMode, Recipe, build_logits_step_fn, build_step_fn
from mlxs.generate.sampling import make_sampler


class _RuntimeTokenizer:
    def __init__(self, mapping: dict[int, str], *, eos_token_id: int) -> None:
        self._mapping = mapping
        self._eos_token_id = eos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def encode(self, text: str) -> list[int]:
        del text
        return [1]

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return self._mapping[token_ids]
        return "".join(self._mapping[token_id] for token_id in token_ids)


class _RuntimeCache:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def state(self) -> Any:
        return None


class _RuntimeModel:
    def __init__(self, predictions: list[int], *, vocab_size: int = 8) -> None:
        self._predictions = predictions
        self._vocab_size = vocab_size
        self.call_index = 0

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
        cache[0].calls += 1
        token = self._predictions[min(self.call_index, len(self._predictions) - 1)]
        self.call_index += 1
        row = mx.where(
            mx.arange(self._vocab_size) == token,
            mx.array(5.0, dtype=mx.float32),
            -mx.arange(self._vocab_size, dtype=mx.float32) - 5.0,
        )
        return mx.broadcast_to(row, (input_ids.shape[0], input_ids.shape[1], self._vocab_size))

    def make_cache(self) -> list[_RuntimeCache]:
        return [_RuntimeCache()]


class _CompileRuntimeModel:
    def __init__(self, predictions: list[int], *, vocab_size: int = 8) -> None:
        self._predictions = predictions
        self._vocab_size = vocab_size
        self.call_index = 0

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        del cache, mask, input_embeddings
        token = self._predictions[min(self.call_index, len(self._predictions) - 1)]
        self.call_index += 1
        row = mx.where(
            mx.arange(self._vocab_size) == token,
            mx.array(5.0, dtype=mx.float32),
            -mx.arange(self._vocab_size, dtype=mx.float32) - 5.0,
        )
        return mx.broadcast_to(row, (input_ids.shape[0], input_ids.shape[1], self._vocab_size))

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]


def _greedy_recipe(*, compile_mode: CompileMode = CompileMode.OFF) -> Recipe:
    return Recipe(
        sampler=make_sampler(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0),
        logits_processors=(),
        has_processors=False,
        processor_context_size=0,
        emit_logprobs=False,
        emit_top_logprobs=False,
        top_logprobs_k=0,
        compile_mode=compile_mode,
    )


def test_seed_stop_does_not_dispatch_next_step() -> None:
    model = _RuntimeModel([0, 7])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=3, temperature=0.0),
        )
    )

    assert [event.token_id for event in events] == [0]
    assert events[0].finish_reason is FinishReason.STOP
    assert model.call_index == 1


def test_length_stop_skips_dispatch_of_following_step() -> None:
    model = _RuntimeModel([7, 5, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 5: "five", 7: "seven"}, eos_token_id=0)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=2, temperature=0.0),
        )
    )

    assert [event.token_id for event in events] == [7, 5]
    assert events[-1].finish_reason is FinishReason.LENGTH
    assert model.call_index == 2


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
    assert model.call_index == 3


def test_v4_uses_seed_eval_and_steady_state_async_submit(monkeypatch: Any) -> None:
    calls: list[tuple[str, int]] = []
    real_async_eval = cast(Any, mx.async_eval)
    real_eval = mx.eval

    def _wrapped_async_eval(*args: Any) -> Any:
        calls.append(("async", len(args)))
        return real_async_eval(*args)

    def _wrapped_eval(*args: Any) -> Any:
        calls.append(("eval", len(args)))
        return real_eval(*args)

    monkeypatch.setattr("mlxs.generate.core.mx.async_eval", _wrapped_async_eval)
    monkeypatch.setattr("mlxs.generate.core.mx.eval", _wrapped_eval)

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
    assert calls == [("async", 1), ("eval", 1), ("async", 1)]


def test_async_eval_runs_outside_stream_context(monkeypatch: Any) -> None:
    state = {"inside_stream": False}
    seen_inside: list[bool] = []
    real_async_eval = cast(Any, mx.async_eval)

    @contextmanager
    def _sentinel_stream(stream: Any) -> Any:
        del stream
        state["inside_stream"] = True
        try:
            yield None
        finally:
            state["inside_stream"] = False

    def _wrapped_async_eval(*args: Any) -> Any:
        seen_inside.append(state["inside_stream"])
        return real_async_eval(*args)

    monkeypatch.setattr("mlxs.generate.core.mx.stream", _sentinel_stream)
    monkeypatch.setattr("mlxs.generate.recipe.mx.stream", _sentinel_stream)
    monkeypatch.setattr("mlxs.generate.core.mx.async_eval", _wrapped_async_eval)

    model = _RuntimeModel([7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)

    list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=2, temperature=0.0),
        )
    )

    assert seen_inside
    assert seen_inside == [False, False]


def test_build_step_fn_keeps_token_shape_and_vocab_logprobs() -> None:
    model = _RuntimeModel([7, 0])
    cache = model.make_cache()
    step_fn = build_step_fn(model, cache, _greedy_recipe(), core_mod._generation_stream)

    next_token, logprobs = step_fn(mx.array([1], dtype=mx.int32))

    assert next_token.shape == (1,)
    assert logprobs.shape == (8,)


def test_compile_decode_supports_compile_mode_and_calls_warmup(monkeypatch: Any) -> None:
    model = _CompileRuntimeModel([7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)
    warmup_calls: list[tuple[Any, Any]] = []
    explicit_calls: list[tuple[int, int]] = []

    def _warmup_stub(model_arg: Any, cache_factory: Any, *, vocab_size: int = 32000) -> None:
        del vocab_size
        warmup_calls.append((model_arg, cache_factory))

    def _explicit_core_stub(
        model_arg: Any,
        cache_arg: Any,
        recipe_arg: Any,
        stream_arg: Any,
        first_logits_arg: Any,
        max_tokens_arg: int,
        clear_cache_interval_arg: int,
        *,
        final_cache_holder: list[list[KVCache] | None] | None = None,
    ) -> Any:
        del model_arg, cache_arg, recipe_arg, stream_arg, first_logits_arg
        explicit_calls.append((max_tokens_arg, clear_cache_interval_arg))
        if final_cache_holder is not None:
            final_cache_holder[:] = [[KVCache()]]
        yield 7, mx.zeros((8,), dtype=mx.float32)
        yield 0, mx.zeros((8,), dtype=mx.float32)

    monkeypatch.setattr("mlxs.generate.warmup", _warmup_stub)
    monkeypatch.setattr("mlxs.generate._decode_steps_explicit_state", _explicit_core_stub)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=2, temperature=0.0),
            compile_decode=True,
        )
    )

    assert [event.token_id for event in events] == [7, 0]
    assert warmup_calls == [(model, model.make_cache)]
    assert explicit_calls == [(2, 256)]


def test_build_step_fn_compile_mode_on_uses_compiled_closure(monkeypatch: Any) -> None:
    model = _RuntimeModel([7, 0])
    cache = model.make_cache()
    compile_calls = {"count": 0}

    def _identity_compile(fn: Any) -> Any:
        compile_calls["count"] += 1
        return fn

    monkeypatch.setattr("mlxs.generate.recipe.mx.compile", _identity_compile)

    step_fn = build_step_fn(
        model,
        cache,
        _greedy_recipe(compile_mode=CompileMode.ON),
        core_mod._generation_stream,
    )

    next_token, logprobs = step_fn(mx.array([1], dtype=mx.int32))

    assert compile_calls["count"] == 1
    assert next_token.shape == (1,)
    assert logprobs.shape == (8,)


def test_group2_repetition_penalty_is_supported_on_public_eager_path() -> None:
    model = _RuntimeModel([7, 6, 5])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 5: "five", 6: "six", 7: "seven"}, eos_token_id=0)
    options = GenerateOptions(
        max_tokens=2,
        temperature=0.8,
        top_p=0.9,
        top_k=4,
        min_p=0.05,
        repetition_penalty=1.2,
        seed=0,
    )

    first = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1],
            options,
        )
    ]
    model = _RuntimeModel([7, 6, 5])
    second = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1],
            options,
        )
    ]

    assert len(first) == 2
    assert first == second


def test_build_logits_step_fn_returns_raw_logits_for_processor_path() -> None:
    model = _RuntimeModel([7, 0])
    cache = model.make_cache()
    recipe = Recipe(
        sampler=make_sampler(temperature=0.8, top_p=0.9, top_k=4, min_p=0.05),
        logits_processors=(object(),),
        has_processors=True,
        processor_context_size=20,
        emit_logprobs=False,
        emit_top_logprobs=False,
        top_logprobs_k=0,
        compile_mode=CompileMode.OFF,
    )

    logits = build_logits_step_fn(model, cache, recipe, core_mod._generation_stream)(
        mx.array([1], dtype=mx.int32)
    )

    assert logits.shape == (8,)
    assert float(logits[7].item()) == pytest.approx(5.0)


def test_group2_repetition_penalty_is_supported_on_public_compile_path(
    monkeypatch: Any,
) -> None:
    model = _CompileRuntimeModel([7, 0])
    model2 = _CompileRuntimeModel([7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)
    warmup_calls: list[tuple[Any, Any]] = []
    captured_recipes: list[Recipe] = []

    def _warmup_stub(model_arg: Any, cache_factory: Any, *, vocab_size: int = 32000) -> None:
        del vocab_size
        warmup_calls.append((model_arg, cache_factory))

    def _explicit_core_stub(
        model_arg: Any,
        cache_arg: Any,
        recipe_arg: Recipe,
        stream_arg: Any,
        first_logits_arg: Any,
        max_tokens_arg: int,
        clear_cache_interval_arg: int,
        *,
        final_cache_holder: list[list[KVCache] | None] | None = None,
    ) -> Any:
        del model_arg, cache_arg, stream_arg, first_logits_arg
        assert max_tokens_arg == 2
        assert clear_cache_interval_arg == 256
        captured_recipes.append(recipe_arg)
        if final_cache_holder is not None:
            final_cache_holder[:] = [[KVCache()]]
        yield 7, mx.zeros((8,), dtype=mx.float32)
        yield 0, mx.zeros((8,), dtype=mx.float32)

    monkeypatch.setattr("mlxs.generate.warmup", _warmup_stub)
    monkeypatch.setattr("mlxs.generate._decode_steps_explicit_state", _explicit_core_stub)
    options = GenerateOptions(
        max_tokens=2,
        temperature=0.8,
        top_p=0.9,
        top_k=4,
        min_p=0.05,
        repetition_penalty=1.2,
        seed=0,
    )

    first = list(generate(model, tokenizer, [1], options, compile_decode=True))
    second = list(
        generate(
            model2,
            tokenizer,
            [1],
            options,
            compile_decode=True,
        )
    )

    assert [event.token_id for event in first] == [7, 0]
    assert [event.token_id for event in first] == [event.token_id for event in second]
    assert warmup_calls == [(model, model.make_cache), (model2, model2.make_cache)]
    assert len(captured_recipes) == 2
    assert all(recipe.has_processors for recipe in captured_recipes)
    assert all(len(recipe.logits_processors) == 1 for recipe in captured_recipes)
    assert all(recipe.processor_context_size == 20 for recipe in captured_recipes)


def test_group3_logprobs_are_supported_on_public_eager_path() -> None:
    model = _RuntimeModel([7, 0])
    token_map = {idx: f"tok{idx}" for idx in range(8)} | {0: "<eos>"}
    tokenizer = _RuntimeTokenizer(token_map, eos_token_id=0)
    options = GenerateOptions(max_tokens=2, temperature=0.0, logprobs=True, top_logprobs=3)

    first = list(generate(model, tokenizer, [1], options))
    second = list(generate(_RuntimeModel([7, 0]), tokenizer, [1], options))

    assert [event.token_id for event in first] == [7, 0]
    assert [event.token_id for event in first] == [event.token_id for event in second]
    assert all(event.logprobs is not None for event in first)
    assert [
        event.logprobs.token_logprob for event in first if event.logprobs is not None
    ] == pytest.approx(
        [
            event.logprobs.token_logprob
            for event in second
            if event.logprobs is not None
        ]
    )
    assert all(
        len(event.logprobs.top_logprobs) == 3
        for event in first
        if event.logprobs is not None
    )


def test_group3_logprobs_are_supported_on_public_compile_path(
    monkeypatch: Any,
) -> None:
    model = _CompileRuntimeModel([7, 0])
    token_map = {idx: f"tok{idx}" for idx in range(8)} | {0: "<eos>"}
    tokenizer = _RuntimeTokenizer(token_map, eos_token_id=0)
    captured_recipes: list[Recipe] = []

    def _explicit_core_stub(
        model_arg: Any,
        cache_arg: Any,
        recipe_arg: Recipe,
        stream_arg: Any,
        first_logits_arg: Any,
        max_tokens_arg: int,
        clear_cache_interval_arg: int,
        *,
        final_cache_holder: list[list[KVCache] | None] | None = None,
    ) -> Any:
        del model_arg, cache_arg, stream_arg, first_logits_arg
        assert max_tokens_arg == 2
        assert clear_cache_interval_arg == 256
        captured_recipes.append(recipe_arg)
        if final_cache_holder is not None:
            final_cache_holder[:] = [[KVCache()]]
        yield 7, mx.array([-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0], dtype=mx.float32)
        yield 0, mx.array([0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0], dtype=mx.float32)

    monkeypatch.setattr("mlxs.generate._decode_steps_explicit_state", _explicit_core_stub)
    monkeypatch.setattr("mlxs.generate.warmup", lambda *args, **kwargs: None)

    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=2, temperature=0.0, logprobs=True, top_logprobs=3),
            compile_decode=True,
        )
    )

    assert [event.token_id for event in events] == [7, 0]
    assert len(captured_recipes) == 1
    assert captured_recipes[0].emit_logprobs is True
    assert captured_recipes[0].emit_top_logprobs is True
    assert captured_recipes[0].top_logprobs_k == 3
    assert events[0].logprobs is not None
    assert events[0].logprobs.token_logprob == pytest.approx(0.0)
    assert [entry.token_id for entry in events[0].logprobs.top_logprobs] == [7, 6, 5]


def test_group4_quantized_kv_is_supported_on_public_eager_path(monkeypatch: Any) -> None:
    model = _CompileRuntimeModel([7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)
    convert_calls: list[tuple[int, int, int]] = []

    def _convert_stub(
        cache_arg: list[KVCache],
        *,
        kv_bits: int = 8,
        kv_group_size: int = 64,
    ) -> list[QuantizedKVCache]:
        convert_calls.append((len(cache_arg), kv_bits, kv_group_size))
        return [QuantizedKVCache(group_size=kv_group_size, bits=kv_bits) for _ in cache_arg]

    monkeypatch.setattr("mlxs.generate.core.convert_to_quantized", _convert_stub)

    final_cache_out: list[list[KVCache]] = []
    events = list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=2, temperature=0.0),
            quantized_kv_start=1,
            kv_bits=8,
            final_cache_out=final_cache_out,
        )
    )

    assert [event.token_id for event in events] == [7, 0]
    assert convert_calls == [(1, 8, 64)]
    assert final_cache_out
    assert isinstance(final_cache_out[0][0], QuantizedKVCache)


def test_group4_quantized_kv_rejects_compile_on_path() -> None:
    model = _CompileRuntimeModel([7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)

    with pytest.raises(
        NotImplementedError,
        match="compile-on does not support delayed quantized KV",
    ):
        list(
            generate(
                model,
                tokenizer,
                [1],
                GenerateOptions(max_tokens=2, temperature=0.0),
                compile_decode=True,
                quantized_kv_start=1,
                kv_bits=8,
            )
        )


def test_compile_on_rejects_non_kvcache_layers() -> None:
    model = _RuntimeModel([7, 0])
    tokenizer = _RuntimeTokenizer({0: "<eos>", 7: "seven"}, eos_token_id=0)

    with pytest.raises(NotImplementedError, match="plain KVCache"):
        list(
            generate(
                model,
                tokenizer,
                [1],
                GenerateOptions(max_tokens=2, temperature=0.0),
                compile_decode=True,
            )
        )
