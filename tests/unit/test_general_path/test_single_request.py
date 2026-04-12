"""Tests for Layer 2 single-request helpers."""

from __future__ import annotations

import mlx.core as mx

from mlxs.general_path.single_request import _build_top_logprobs


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
