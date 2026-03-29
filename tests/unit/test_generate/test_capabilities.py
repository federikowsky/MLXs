"""Tests for internal decode capability resolution and integration."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from mlxs._errors import InvalidPromptError
from mlxs._types import GenerateOptions
from mlxs.cache import ArraysCache, CacheList, ChunkedKVCache, KVCache
from mlxs.generate import generate
from mlxs.generate.capabilities import (
    CacheTopology,
    PrefillSyncStrategy,
    resolve_decode_capabilities,
)
from mlxs.generate.prefill import chunked_prefill


class _ModelWithEmbeddings:
    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        del cache, input_embeddings
        return mx.zeros((input_ids.shape[0], input_ids.shape[1], 8))


class _ModelWithoutEmbeddings:
    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Any = None,
    ) -> mx.array:
        del cache
        return mx.zeros((input_ids.shape[0], input_ids.shape[1], 8))


class _GenerateTokenizer:
    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        raise AssertionError("Tests pass token ids directly.")

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return str(token_ids)
        return "".join(str(token_id) for token_id in token_ids)


class _GenerateModel:
    def __init__(self) -> None:
        self._cache = [ArraysCache(size=1)]

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Any = None,
    ) -> mx.array:
        del cache
        vocab_size = 4
        return mx.zeros((input_ids.shape[0], input_ids.shape[1], vocab_size))

    def make_cache(self) -> list[Any]:
        return self._cache


def test_resolve_decode_capabilities_for_full_kv_cache() -> None:
    capabilities = resolve_decode_capabilities(_ModelWithEmbeddings(), [KVCache()])

    assert capabilities.cache_topology is CacheTopology.KV
    assert capabilities.supports_delayed_kv_quantization is True
    assert capabilities.prefill_sync_strategy is PrefillSyncStrategy.CACHE_STATE
    assert capabilities.supports_prefill_input_embeddings is True


def test_resolve_decode_capabilities_for_arrays_cache() -> None:
    capabilities = resolve_decode_capabilities(
        _ModelWithoutEmbeddings(),
        [ArraysCache(size=1)],
    )

    assert capabilities.cache_topology is CacheTopology.STATE
    assert capabilities.supports_delayed_kv_quantization is False
    assert capabilities.prefill_sync_strategy is PrefillSyncStrategy.MODEL_OUTPUT
    assert capabilities.supports_prefill_input_embeddings is False


def test_resolve_decode_capabilities_for_hybrid_cache_list() -> None:
    capabilities = resolve_decode_capabilities(
        _ModelWithoutEmbeddings(),
        [CacheList(ArraysCache(size=1), KVCache())],
    )

    assert capabilities.cache_topology is CacheTopology.HYBRID
    assert capabilities.supports_delayed_kv_quantization is False
    assert capabilities.prefill_sync_strategy is PrefillSyncStrategy.MODEL_OUTPUT


def test_resolve_decode_capabilities_for_chunked_kv_cache() -> None:
    capabilities = resolve_decode_capabilities(
        _ModelWithoutEmbeddings(),
        [ChunkedKVCache(16)],
    )

    assert capabilities.cache_topology is CacheTopology.KV
    assert capabilities.supports_delayed_kv_quantization is False
    assert capabilities.prefill_sync_strategy is PrefillSyncStrategy.MODEL_OUTPUT


def test_chunked_prefill_syncs_on_model_output_when_cache_state_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ModelWithoutEmbeddings()
    cache = [ArraysCache(size=1)]
    eval_calls: list[tuple[Any, ...]] = []

    def fake_eval(*args: Any) -> None:
        eval_calls.append(args)

    monkeypatch.setattr("mlxs.generate.prefill.mx.eval", fake_eval)

    chunked_prefill(
        model,
        mx.array([1, 2, 3]),
        cache,
        prefill_step_size=2,
        capabilities=resolve_decode_capabilities(model, cache),
    )

    assert len(eval_calls) == 1
    assert len(eval_calls[0]) == 1
    assert hasattr(eval_calls[0][0], "shape")


def test_generate_rejects_delayed_quantized_kv_on_unsupported_cache_topology() -> None:
    with pytest.raises(
        ValueError,
        match="Delayed quantized KV requires full-precision KV caches",
    ):
        list(
            generate(
                _GenerateModel(),
                _GenerateTokenizer(),
                [1],
                GenerateOptions(max_tokens=1, temperature=0),
                quantized_kv_start=1,
                kv_bits=4,
            )
        )


def test_generate_rejects_input_embeddings_when_model_does_not_support_them() -> None:
    with pytest.raises(InvalidPromptError, match="does not support input_embeddings"):
        list(
            generate(
                _GenerateModel(),
                _GenerateTokenizer(),
                [1],
                GenerateOptions(max_tokens=1, temperature=0),
                input_embeddings=mx.zeros((1, 8)),
            )
        )
