"""Tests for shared value types (_types.py).

Patterns: happy path, boundary, corner cases, alternate flows.
"""

from __future__ import annotations

import pytest

from mlxs._types import (
    FinishReason,
    GenerateOptions,
    GenerateResult,
    MemoryCeilingPolicy,
    ModelMode,
    PaddingSide,
    RequestMetrics,
    StreamPolicy,
    TokenEvent,
    TokenLogprobs,
    ToolCallResult,
    TopLogprob,
)

# =============================================================================
# Enums
# =============================================================================


class TestEnums:
    def test_finish_reason_values(self) -> None:
        assert len(FinishReason) == 4
        assert FinishReason.STOP.name == "STOP"
        assert FinishReason.LENGTH.name == "LENGTH"
        assert FinishReason.TOOL_CALLS.name == "TOOL_CALLS"
        assert FinishReason.CANCELLED.name == "CANCELLED"

    def test_stream_policy_values(self) -> None:
        assert StreamPolicy.SINGLE.value == "single"
        assert StreamPolicy.OVERLAP.value == "overlap"

    def test_memory_ceiling_policy_values(self) -> None:
        assert MemoryCeilingPolicy.TRIM_CACHE.value == "trim_cache"
        assert MemoryCeilingPolicy.REJECT_ONLY.value == "reject_only"
        assert MemoryCeilingPolicy.SHUTDOWN.value == "shutdown"

    def test_model_mode_values(self) -> None:
        assert ModelMode.TEXT.value == "text"
        assert ModelMode.MULTIMODAL.value == "multimodal"
        assert ModelMode.AUTO.value == "auto"

    def test_padding_side_values(self) -> None:
        assert PaddingSide.LEFT.value == "left"
        assert PaddingSide.RIGHT.value == "right"


# =============================================================================
# TokenEvent
# =============================================================================


class TestTokenEvent:
    def test_defaults(self) -> None:
        e = TokenEvent(token_id=1, text="hi")
        assert e.finish_reason is None
        assert e.logprobs is None
        assert e.prompt_tokens == 0
        assert e.generation_tokens == 0
        assert e.timestamp > 0

    def test_all_fields(self) -> None:
        lp = TokenLogprobs(token_logprob=-0.5)
        e = TokenEvent(
            token_id=42,
            text="word",
            finish_reason=FinishReason.STOP,
            logprobs=lp,
            prompt_tokens=10,
            generation_tokens=5,
        )
        assert e.token_id == 42
        assert e.text == "word"
        assert e.finish_reason == FinishReason.STOP
        assert e.logprobs.token_logprob == -0.5

    def test_mutable(self) -> None:
        """TokenEvent is mutable (not frozen) for hot path efficiency."""
        e = TokenEvent(token_id=1, text="a")
        e.finish_reason = FinishReason.LENGTH
        assert e.finish_reason == FinishReason.LENGTH

    def test_slots(self) -> None:
        e = TokenEvent(token_id=1, text="x")
        with pytest.raises(AttributeError):
            e.nonexistent = True  # type: ignore[attr-defined]


# =============================================================================
# GenerateOptions
# =============================================================================


class TestGenerateOptions:
    def test_defaults(self) -> None:
        opts = GenerateOptions()
        assert opts.max_tokens == 512
        assert opts.temperature == 1.0
        assert opts.top_p == 1.0
        assert opts.top_k == 0
        assert opts.min_p == 0.0
        assert opts.seed is None
        assert opts.stop_sequences == ()
        assert opts.extra_eos_token_ids == ()
        assert opts.repetition_penalty == 1.0
        assert opts.logprobs is False
        assert opts.top_logprobs == 0
        assert opts.stream is True

    def test_frozen(self) -> None:
        opts = GenerateOptions()
        with pytest.raises(AttributeError):
            opts.max_tokens = 100  # type: ignore[misc]

    def test_custom_values(self) -> None:
        opts = GenerateOptions(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            min_p=0.05,
            seed=42,
            stop_sequences=("<|end|>",),
            logprobs=True,
            top_logprobs=5,
        )
        assert opts.max_tokens == 100
        assert opts.temperature == 0.7
        assert opts.seed == 42
        assert opts.logprobs is True
        assert opts.top_logprobs == 5


class TestGenerateOptionsBoundary:
    def test_temperature_zero_greedy(self) -> None:
        opts = GenerateOptions(temperature=0.0)
        assert opts.temperature == 0.0

    def test_max_tokens_zero(self) -> None:
        opts = GenerateOptions(max_tokens=0)
        assert opts.max_tokens == 0

    def test_top_logprobs_max(self) -> None:
        opts = GenerateOptions(top_logprobs=20)
        assert opts.top_logprobs == 20


# =============================================================================
# TopLogprob / TokenLogprobs
# =============================================================================


class TestLogprobTypes:
    def test_top_logprob_frozen(self) -> None:
        tlp = TopLogprob(token_id=1, token="the", logprob=-0.5)
        with pytest.raises(AttributeError):
            tlp.logprob = -1.0  # type: ignore[misc]

    def test_token_logprobs_defaults(self) -> None:
        lp = TokenLogprobs(token_logprob=-0.3)
        assert lp.top_logprobs == ()

    def test_token_logprobs_with_top(self) -> None:
        lp = TokenLogprobs(
            token_logprob=-0.3,
            top_logprobs=(
                TopLogprob(token_id=1, token="a", logprob=-0.3),
                TopLogprob(token_id=2, token="b", logprob=-1.0),
            ),
        )
        assert len(lp.top_logprobs) == 2
        assert lp.top_logprobs[0].token == "a"


# =============================================================================
# ToolCallResult
# =============================================================================


class TestToolCallResult:
    def test_frozen(self) -> None:
        r = ToolCallResult(id="1", name="f", arguments="{}")
        with pytest.raises(AttributeError):
            r.name = "g"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ToolCallResult(id="1", name="f", arguments="{}")
        b = ToolCallResult(id="1", name="f", arguments="{}")
        assert a == b

    def test_inequality(self) -> None:
        a = ToolCallResult(id="1", name="f", arguments="{}")
        b = ToolCallResult(id="2", name="f", arguments="{}")
        assert a != b


# =============================================================================
# GenerateResult
# =============================================================================


class TestGenerateResult:
    def test_frozen(self) -> None:
        r = GenerateResult(
            text="hello",
            token_ids=(1, 2),
            prompt_tokens=5,
            generation_tokens=2,
            finish_reason=FinishReason.STOP,
        )
        with pytest.raises(AttributeError):
            r.text = "bye"  # type: ignore[misc]

    def test_defaults(self) -> None:
        r = GenerateResult(
            text="x",
            token_ids=(1,),
            prompt_tokens=1,
            generation_tokens=1,
            finish_reason=FinishReason.LENGTH,
        )
        assert r.tool_calls == ()
        assert r.tokens_per_second == 0.0
        assert r.time_to_first_token == 0.0


# =============================================================================
# RequestMetrics
# =============================================================================


class TestRequestMetrics:
    def test_defaults(self) -> None:
        m = RequestMetrics()
        assert m.decode_tokens_per_second == 0.0
        assert m.prefill_tokens_per_second == 0.0
        assert m.prompt_cache_hit is False

    def test_custom(self) -> None:
        m = RequestMetrics(
            decode_tokens_per_second=150.5,
            prompt_tokens=100,
            generation_tokens=50,
            prompt_cache_hit=True,
        )
        assert m.decode_tokens_per_second == 150.5
        assert m.prompt_cache_hit is True
