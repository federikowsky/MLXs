"""Tests for merge_embeddings (§7.4, FR12)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlxs._errors import InvalidPromptError
from mlxs.models.vision import merge_embeddings


def test_merge_replaces_placeholders() -> None:
    """Placeholder positions are replaced with media embeddings."""
    B, T, D = 1, 5, 4
    placeholder_id = 99

    input_ids = mx.array([[1, 99, 99, 2, 3]])
    text_embeds = mx.ones((B, T, D))
    media_embeds = mx.full((2, D), 42.0)

    result = merge_embeddings(text_embeds, media_embeds, input_ids, placeholder_id)

    assert result.shape == (B, T, D)
    # Positions 1, 2 should have media value 42.0
    assert float(result[0, 1, 0].item()) == 42.0
    assert float(result[0, 2, 0].item()) == 42.0
    # Other positions should retain original value 1.0
    assert float(result[0, 0, 0].item()) == 1.0
    assert float(result[0, 3, 0].item()) == 1.0
    assert float(result[0, 4, 0].item()) == 1.0


def test_merge_no_placeholders() -> None:
    """When no placeholders, output equals input text embeddings."""
    B, T, D = 1, 3, 4
    input_ids = mx.array([[1, 2, 3]])
    text_embeds = mx.ones((B, T, D))
    media_embeds = mx.zeros((0, D))

    result = merge_embeddings(text_embeds, media_embeds, input_ids, 99)
    assert mx.allclose(result, text_embeds)


def test_merge_count_mismatch_raises() -> None:
    """Mismatch between placeholder count and media embeddings raises."""
    B, T, D = 1, 5, 4
    input_ids = mx.array([[1, 99, 99, 99, 3]])  # 3 placeholders
    text_embeds = mx.ones((B, T, D))
    media_embeds = mx.zeros((2, D))  # 2 media tokens != 3 placeholders

    with pytest.raises(InvalidPromptError, match="media tokens"):
        merge_embeddings(text_embeds, media_embeds, input_ids, 99)


def test_merge_output_shape() -> None:
    """Output shape matches input text_embeddings shape."""
    B, T, D = 1, 10, 8
    placeholder_id = 50
    n_media = 4
    input_ids = mx.array([[50, 1, 50, 2, 50, 3, 50, 4, 5, 6]])
    text_embeds = mx.ones((B, T, D))
    media_embeds = mx.zeros((n_media, D))

    result = merge_embeddings(text_embeds, media_embeds, input_ids, placeholder_id)
    assert result.shape == (B, T, D)
