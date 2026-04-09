"""Tests for Decode Engine V4 profiling diagnostics."""

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
        del text
        return [1]

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return str(token_ids)
        return "".join(str(token_id) for token_id in token_ids)


class _ProfileCache:
    @property
    def state(self) -> Any:
        return None


class _ProfileModel:
    def __init__(self, predictions: list[int], *, vocab_size: int = 8) -> None:
        self._predictions = predictions
        self._vocab_size = vocab_size
        self.calls = 0

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
        token = self._predictions[min(self.calls, len(self._predictions) - 1)]
        self.calls += 1
        row = mx.where(
            mx.arange(self._vocab_size) == token,
            mx.array(0.0),
            mx.array(-100.0),
        )
        return mx.broadcast_to(row, (input_ids.shape[0], input_ids.shape[1], self._vocab_size))

    def make_cache(self) -> list[_ProfileCache]:
        return [_ProfileCache()]


def test_decode_profile_logs_v4_summary_when_enabled(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    monkeypatch.setenv("MLXS_DECODE_PROFILE", "1")
    monkeypatch.delenv("MLXS_DECODE_PROFILE_DETAIL", raising=False)
    caplog.set_level(logging.WARNING, logger="mlxs.generate.profile")

    model = _ProfileModel([7, 5, 0])
    tokenizer = _ProfileTokenizer()

    list(generate(model, tokenizer, [1], GenerateOptions(max_tokens=3, temperature=0.0)))

    joined = "\n".join(record.message for record in caplog.records)
    assert "[MLXS_DECODE_PROFILE]" in joined
    assert "detail_mode=False" in joined
    assert "prefill_wall_s=" in joined
    assert "first_token_wall_s=" in joined
    assert "per_step_wall_s:" in joined
    assert "step_fn_wall_s=n/a" in joined


def test_decode_profile_logs_v4_detail_metrics_when_enabled(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    monkeypatch.setenv("MLXS_DECODE_PROFILE", "1")
    monkeypatch.setenv("MLXS_DECODE_PROFILE_DETAIL", "1")
    caplog.set_level(logging.WARNING, logger="mlxs.generate.profile")

    model = _ProfileModel([7, 5, 0])
    tokenizer = _ProfileTokenizer()

    list(generate(model, tokenizer, [1], GenerateOptions(max_tokens=3, temperature=0.0)))

    joined = "\n".join(record.message for record in caplog.records)
    assert "detail_mode=True" in joined
    assert "async_eval_wall_s:" in joined
    assert "item_wait_wall_s:" in joined
    assert "tokenizer_wall_s:" in joined
    assert "stop_check_wall_s:" in joined
    assert "event_build_wall_s:" in joined
    assert "boundary proxy note" in joined
