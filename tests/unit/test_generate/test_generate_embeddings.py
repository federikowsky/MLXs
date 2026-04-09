"""Tests for generate() with input_embeddings (§7.4, FR12)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from mlxs._errors import InvalidPromptError
from mlxs._types import GenerateOptions
from mlxs.generate import generate


class _FakeTokenizer:
    """Minimal tokenizer for generate() tests."""

    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3, 4]

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return "x"
        return "x" * len(token_ids)


class _FakeModel:
    """Model that checks input_embeddings usage."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 8) -> None:
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self.received_embeddings: list[bool] = []

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Any = None,
        mask: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        self.received_embeddings.append(input_embeddings is not None)
        B, T = input_ids.shape
        logits = mx.zeros((B, T, self._vocab_size))
        return logits

    @property
    def num_layers(self) -> int:
        return 1

    def make_cache(self) -> list[Any]:
        return [_FakeCache()]


class _FakeCache:
    """Minimal cache mock."""

    @property
    def state(self) -> Any:
        return None


def test_generate_with_none_embeddings() -> None:
    """generate() with input_embeddings=None works normally."""
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    opts = GenerateOptions(max_tokens=1, temperature=0.0)
    events = list(generate(model, tokenizer, [1, 2, 3], opts))
    assert len(events) == 1
    # Prefill should NOT have received embeddings
    assert model.received_embeddings[0] is False


def test_generate_with_embeddings() -> None:
    """generate() with valid input_embeddings passes them to the model."""
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    opts = GenerateOptions(max_tokens=1, temperature=0.0)
    prompt = [1, 2, 3]
    embeds = mx.zeros((3, 8))  # (T, D) matching prompt length
    events = list(generate(model, tokenizer, prompt, opts, input_embeddings=embeds))
    assert len(events) == 1
    # Prefill should have received embeddings
    assert model.received_embeddings[0] is True


def test_generate_embeddings_wrong_ndim() -> None:
    """generate() raises InvalidPromptError for wrong ndim."""
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    opts = GenerateOptions(max_tokens=1, temperature=0.0)
    embeds = mx.zeros((1, 3, 8))  # 3-D, should be 2-D
    with pytest.raises(InvalidPromptError, match="2-D"):
        list(generate(model, tokenizer, [1, 2, 3], opts, input_embeddings=embeds))


def test_generate_embeddings_length_mismatch() -> None:
    """generate() raises InvalidPromptError when embed length != prompt length."""
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    opts = GenerateOptions(max_tokens=1, temperature=0.0)
    embeds = mx.zeros((5, 8))  # Length 5 != prompt length 3
    with pytest.raises(InvalidPromptError, match="length"):
        list(generate(model, tokenizer, [1, 2, 3], opts, input_embeddings=embeds))
