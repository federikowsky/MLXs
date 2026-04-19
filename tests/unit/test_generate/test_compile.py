"""Tests for compile eligibility helpers."""

from __future__ import annotations

from mlxs.generate.compile import (
    COMPILED_DECODE_PROMPT_TOKEN_LIMIT,
    compile_decode_eligible,
)


def test_compile_decode_eligible_allows_threshold_and_below() -> None:
    assert compile_decode_eligible(prompt_token_count=1) is True
    assert (
        compile_decode_eligible(prompt_token_count=COMPILED_DECODE_PROMPT_TOKEN_LIMIT) is True
    )


def test_compile_decode_eligible_rejects_above_threshold() -> None:
    assert (
        compile_decode_eligible(prompt_token_count=COMPILED_DECODE_PROMPT_TOKEN_LIMIT + 1)
        is False
    )
