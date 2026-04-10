"""Tests for Layer 1 prefill."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from mlxs.runtime_core.policy import CoreExecutionPolicy
from mlxs.runtime_core.prefill import run_prefill
from mlxs.runtime_core.state import CoreState


class _TrackingCache:
    @property
    def state(self) -> Any:
        return None


class _TrackingModel:
    def __init__(self, vocab_size: int = 32) -> None:
        self.calls: list[dict[str, Any]] = []
        self._vocab_size = vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Any = None,
        input_embeddings: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        del cache, mask
        self.calls.append(
            {
                "input_ids_shape": tuple(int(dim) for dim in input_ids.shape),
                "has_embeddings": input_embeddings is not None,
                "embeddings_shape": (
                    tuple(int(dim) for dim in input_embeddings.shape)
                    if input_embeddings is not None
                    else None
                ),
            }
        )
        batch, length = input_ids.shape
        return mx.zeros((batch, length, self._vocab_size))


def test_prefill_without_embeddings() -> None:
    model = _TrackingModel()
    state = CoreState.adopt([_TrackingCache()])
    logits = run_prefill(
        model,
        mx.array([1, 2, 3, 4, 5]),
        state,
        execution=CoreExecutionPolicy(),
        prefill_step_size=3,
    )
    assert logits.shape == (1, 32)
    assert state.prompt_tokens == 5
    assert all(not call["has_embeddings"] for call in model.calls)


def test_prefill_with_embeddings() -> None:
    model = _TrackingModel()
    state = CoreState.adopt([_TrackingCache()])
    logits = run_prefill(
        model,
        mx.array([1, 2, 3, 4, 5]),
        state,
        execution=CoreExecutionPolicy(),
        prefill_step_size=3,
        input_embeddings=mx.zeros((5, 8)),
    )
    assert logits.shape == (1, 32)
    assert state.prompt_tokens == 5
    assert all(call["has_embeddings"] for call in model.calls)


def test_prefill_embedding_chunk_sizes() -> None:
    model = _TrackingModel()
    state = CoreState.adopt([_TrackingCache()])
    run_prefill(
        model,
        mx.array([1, 2, 3, 4, 5, 6, 7]),
        state,
        execution=CoreExecutionPolicy(),
        prefill_step_size=3,
        input_embeddings=mx.zeros((7, 16)),
    )
    for call in model.calls:
        if call["has_embeddings"]:
            assert call["embeddings_shape"][0] == call["input_ids_shape"][0]
            assert call["embeddings_shape"][1] == call["input_ids_shape"][1]


def test_prefill_single_token_with_embeddings() -> None:
    model = _TrackingModel()
    state = CoreState.adopt([_TrackingCache()])
    logits = run_prefill(
        model,
        mx.array([42]),
        state,
        execution=CoreExecutionPolicy(),
        input_embeddings=mx.zeros((1, 8)),
    )
    assert logits.shape == (1, 32)
    assert len(model.calls) == 1
    assert model.calls[0]["has_embeddings"] is True


def test_prefill_rejects_wrong_embedding_ndim() -> None:
    model = _TrackingModel()
    state = CoreState.adopt([_TrackingCache()])
    with pytest.raises(ValueError, match="2-D"):
        run_prefill(
            model,
            mx.array([1, 2, 3]),
            state,
            execution=CoreExecutionPolicy(),
            input_embeddings=mx.zeros((1, 3, 8)),
        )


def test_prefill_rejects_embedding_length_mismatch() -> None:
    model = _TrackingModel()
    state = CoreState.adopt([_TrackingCache()])
    with pytest.raises(ValueError, match="length"):
        run_prefill(
            model,
            mx.array([1, 2, 3]),
            state,
            execution=CoreExecutionPolicy(),
            input_embeddings=mx.zeros((5, 8)),
        )
