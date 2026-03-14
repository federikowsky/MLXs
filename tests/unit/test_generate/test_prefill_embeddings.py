"""Tests for chunked_prefill with input_embeddings (§7.4)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlxs.generate.prefill import chunked_prefill


class _TrackingCache:
    """Minimal cache that tracks state."""

    @property
    def state(self) -> Any:
        return None


class _TrackingModel:
    """Model that records how it was called for prefill verification."""

    def __init__(self, vocab_size: int = 32) -> None:
        self.calls: list[dict[str, Any]] = []
        self._vocab_size = vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        self.calls.append({
            "input_ids_shape": input_ids.shape,
            "has_embeddings": input_embeddings is not None,
            "embeddings_shape": (
                input_embeddings.shape if input_embeddings is not None else None
            ),
        })
        B, T = input_ids.shape
        return mx.zeros((B, T, self._vocab_size))


def test_prefill_without_embeddings() -> None:
    """Prefill without embeddings works normally (baseline)."""
    model = _TrackingModel()
    tokens = mx.array([1, 2, 3, 4, 5])
    cache = [_TrackingCache()]
    logits = chunked_prefill(model, tokens, cache, prefill_step_size=3)
    assert logits.shape == (1, 32)
    # All calls should have no embeddings
    assert all(not c["has_embeddings"] for c in model.calls)


def test_prefill_with_embeddings() -> None:
    """Prefill with embeddings passes sliced embeddings to model."""
    model = _TrackingModel()
    tokens = mx.array([1, 2, 3, 4, 5])
    embeds = mx.zeros((5, 8))
    cache = [_TrackingCache()]
    logits = chunked_prefill(
        model, tokens, cache, prefill_step_size=3, input_embeddings=embeds,
    )
    assert logits.shape == (1, 32)
    # All calls should have embeddings
    assert all(c["has_embeddings"] for c in model.calls)


def test_prefill_embedding_chunk_sizes() -> None:
    """Embeddings are sliced consistently with token chunks."""
    model = _TrackingModel()
    tokens = mx.array([1, 2, 3, 4, 5, 6, 7])
    D = 16
    embeds = mx.zeros((7, D))
    cache = [_TrackingCache()]

    chunked_prefill(
        model, tokens, cache, prefill_step_size=3, input_embeddings=embeds,
    )

    # With 7 tokens and step=3: chunks should be [3, 3, 1]
    # But last token is separate, so: [3, 2] + [1] (last) = 3 calls if 6 remain
    # Actually: total=7, first chunk=3 (offset 0-2), remaining=3,
    # second chunk=3 (offset 3-5), remaining=0, then last token (offset 6)
    for call in model.calls:
        if call["has_embeddings"]:
            # Embedding batch dim should match input_ids batch dim
            assert call["embeddings_shape"][0] == call["input_ids_shape"][0]
            # Embedding seq dim should match input_ids seq dim
            assert call["embeddings_shape"][1] == call["input_ids_shape"][1]


def test_prefill_single_token_with_embeddings() -> None:
    """Prefill with a single token and embeddings."""
    model = _TrackingModel()
    tokens = mx.array([42])
    embeds = mx.zeros((1, 8))
    cache = [_TrackingCache()]
    logits = chunked_prefill(
        model, tokens, cache, input_embeddings=embeds,
    )
    assert logits.shape == (1, 32)
    assert len(model.calls) == 1
    assert model.calls[0]["has_embeddings"] is True
