"""Tests for env-gated decode profiling diagnostics."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from mlxs._types import GenerateOptions
from mlxs.generate import generate


class _ProfileTokenizer:
    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return str(token_ids)
        return "".join(str(token_id) for token_id in token_ids)


class _ProfileCache:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def state(self) -> Any:
        return None


class _ProfileModel:
    def __init__(self, predictions: list[int], *, vocab_size: int = 8) -> None:
        self._predictions = predictions
        self._vocab_size = vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[_ProfileCache] | None = None,
        mask: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        del mask, input_embeddings
        assert cache is not None
        idx = cache[0].calls
        cache[0].calls += 1
        token = self._predictions[min(idx, len(self._predictions) - 1)]
        row = mx.where(
            mx.arange(self._vocab_size) == token,
            mx.array(0.0),
            mx.array(-100.0),
        )
        return mx.broadcast_to(row, (input_ids.shape[0], input_ids.shape[1], self._vocab_size))

    def make_cache(self) -> list[_ProfileCache]:
        return [_ProfileCache()]


def test_decode_profile_logs_summary_when_enabled(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    monkeypatch.setenv("MLXS_DECODE_PROFILE", "1")
    model = _ProfileModel([7, 5, 0])
    tokenizer = _ProfileTokenizer()
    caplog.set_level(logging.WARNING, logger="mlxs.generate.profile")

    list(generate(model, tokenizer, [1, 2, 3], GenerateOptions(max_tokens=3, temperature=0.0)))

    joined = "\n".join(record.message for record in caplog.records)
    assert "[MLXS_DECODE_PROFILE]" in joined
    assert "forward_call=" in joined
    assert "post_forward_tensor=" in joined
    assert "sync_eval=" in joined
    assert "host_materialize=" in joined
    assert "mutation_boundary=" in joined
    assert "compiled_forward_calls=0" in joined
    assert "uncompiled_forward_calls=1" in joined


def test_decode_profile_logs_async_boundary_when_enabled(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    monkeypatch.setenv("MLXS_DECODE_PROFILE", "1")
    monkeypatch.setenv("MLXS_DECODE_ASYNC_EVAL", "1")
    caplog.set_level(logging.WARNING, logger="mlxs.generate.profile")

    model = _ProfileModel([7, 5, 0])
    tokenizer = _ProfileTokenizer()
    list(generate(model, tokenizer, [1], GenerateOptions(max_tokens=2, temperature=0.0)))

    joined = "\n".join(record.message for record in caplog.records)
    assert "boundary_mode=async" in joined
    assert "async_enqueue=" in joined


def test_decode_profile_logs_compile_fallback(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    monkeypatch.setenv("MLXS_DECODE_PROFILE", "1")
    caplog.set_level(logging.WARNING, logger="mlxs.generate.profile")

    def _boom(model: Any, cache: list[Any]) -> Any:
        raise RuntimeError("compile exploded")

    monkeypatch.setattr("mlxs.generate.compile.make_compiled_step", _boom)

    model = _ProfileModel([7, 5, 0])
    tokenizer = _ProfileTokenizer()
    list(
        generate(
            model,
            tokenizer,
            [1, 2, 3],
            GenerateOptions(max_tokens=3, temperature=0.0),
            compile_decode=True,
        )
    )

    joined = "\n".join(record.message for record in caplog.records)
    assert "compile_build_attempted=True" in joined
    assert "compiled_forward_available=False" in joined
    assert "compile_fallback_to_uncompiled=True" in joined
    assert "RuntimeError: compile exploded" in joined


def test_decode_profile_logs_cache_replacement_under_compiled_forward(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    monkeypatch.setenv("MLXS_DECODE_PROFILE", "1")
    caplog.set_level(logging.WARNING, logger="mlxs.generate.profile")

    def _fake_make_compiled_step(model: _ProfileModel, cache: list[_ProfileCache]) -> Any:
        return lambda input_ids: model(input_ids, cache=cache)

    def _fake_convert_to_quantized(
        cache: list[_ProfileCache],
        *,
        kv_bits: int,
        kv_group_size: int,
    ) -> list[_ProfileCache]:
        del kv_bits, kv_group_size
        return [_ProfileCache() for _ in cache]

    monkeypatch.setattr("mlxs.generate.compile.make_compiled_step", _fake_make_compiled_step)
    monkeypatch.setattr("mlxs.cache.convert_to_quantized", _fake_convert_to_quantized)

    model = _ProfileModel([7, 5, 6, 7])
    tokenizer = _ProfileTokenizer()
    list(
        generate(
            model,
            tokenizer,
            [1, 2, 3],
            GenerateOptions(max_tokens=3, temperature=0.0),
            compile_decode=True,
            quantized_kv_start=1,
            kv_bits=8,
        )
    )

    joined = "\n".join(record.message for record in caplog.records)
    assert "compiled_forward_available=True" in joined
    assert "first_compiled_use_forward_step=1" in joined
    assert "first_compiled_use_generation_token=2" in joined
    assert "cache_replacement_events=1" in joined
    assert "cache_replacement_while_compiled_active=1" in joined
    assert "compile_rebind_attempts=1" in joined
    assert "compile_rebind_successes=1" in joined
    assert "compile_rebind_failures=0" in joined


def test_decode_profile_logs_compile_rebind_fallback_to_uncompiled(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    monkeypatch.setenv("MLXS_DECODE_PROFILE", "1")
    caplog.set_level(logging.WARNING, logger="mlxs.generate.profile")
    build_count = 0

    def _fake_make_compiled_step(model: _ProfileModel, cache: list[_ProfileCache]) -> Any:
        nonlocal build_count
        build_count += 1
        if build_count == 1:
            return lambda input_ids: model(input_ids, cache=cache)
        raise RuntimeError("rebind exploded")

    def _fake_convert_to_quantized(
        cache: list[_ProfileCache],
        *,
        kv_bits: int,
        kv_group_size: int,
    ) -> list[_ProfileCache]:
        del kv_bits, kv_group_size
        return [_ProfileCache() for _ in cache]

    monkeypatch.setattr("mlxs.generate.compile.make_compiled_step", _fake_make_compiled_step)
    monkeypatch.setattr("mlxs.cache.convert_to_quantized", _fake_convert_to_quantized)

    model = _ProfileModel([7, 5, 0])
    tokenizer = _ProfileTokenizer()
    list(
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

    joined = "\n".join(record.message for record in caplog.records)
    assert build_count == 2
    assert "compile_rebind_attempts=1" in joined
    assert "compile_rebind_failures=1" in joined
    assert "compile_rebind_fallback_to_uncompiled=1" in joined
    assert "compile_rebind_last_error=RuntimeError: rebind exploded" in joined


def test_decode_sync_eval_only_materializes_token_when_logprobs_disabled(
    monkeypatch: Any,
) -> None:
    observed_arg_counts: list[int] = []
    real_eval = mx.eval

    def _wrapped_eval(*args: Any) -> Any:
        observed_arg_counts.append(len(args))
        return real_eval(*args)

    monkeypatch.setattr("mlxs.generate.runtime.mx.eval", _wrapped_eval)

    model = _ProfileModel([7, 5])
    tokenizer = _ProfileTokenizer()
    list(generate(model, tokenizer, [1], GenerateOptions(max_tokens=1, temperature=0.0)))

    assert observed_arg_counts[0] == 1


def test_decode_sync_eval_materializes_logprobs_when_requested(
    monkeypatch: Any,
) -> None:
    observed_arg_counts: list[int] = []
    real_eval = mx.eval

    def _wrapped_eval(*args: Any) -> Any:
        observed_arg_counts.append(len(args))
        return real_eval(*args)

    monkeypatch.setattr("mlxs.generate.runtime.mx.eval", _wrapped_eval)

    model = _ProfileModel([7, 5])
    tokenizer = _ProfileTokenizer()
    list(
        generate(
            model,
            tokenizer,
            [1],
            GenerateOptions(max_tokens=1, temperature=0.0, logprobs=True),
        )
    )

    assert observed_arg_counts[0] == 2
