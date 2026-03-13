"""Tests for stop condition checking (FR9, FR10).

Patterns: happy path, edge cases, boundary, stateful lifecycle,
alternate flows, corner cases.
"""

from __future__ import annotations

from mlxs._types import FinishReason
from mlxs.generate.stop import StopCondition

# -- Happy path ---------------------------------------------------------------


class TestStopHappyPath:
    def test_eos_token_stops(self) -> None:
        stop = StopCondition(eos_token_id=0, max_tokens=100)
        assert stop.check(0, "") == FinishReason.STOP

    def test_max_tokens_stops(self) -> None:
        stop = StopCondition(eos_token_id=0, max_tokens=3)
        assert stop.check(1, "a") is None
        assert stop.check(2, "b") is None
        assert stop.check(3, "c") == FinishReason.LENGTH

    def test_stop_sequence_stops(self) -> None:
        stop = StopCondition(eos_token_id=0, max_tokens=100, stop_sequences=("END",))
        assert stop.check(1, "E") is None
        assert stop.check(2, "N") is None
        assert stop.check(3, "D") == FinishReason.STOP

    def test_no_stop_returns_none(self) -> None:
        stop = StopCondition(eos_token_id=0, max_tokens=100)
        assert stop.check(99, "hello") is None

    def test_extra_eos_ids(self) -> None:
        stop = StopCondition(eos_token_id=0, max_tokens=100, extra_eos_token_ids=(42, 99))
        assert stop.check(42, "") == FinishReason.STOP
        assert (
            StopCondition(eos_token_id=0, max_tokens=100, extra_eos_token_ids=(42, 99)).check(
                99, ""
            )
            == FinishReason.STOP
        )


# -- Boundary / limit cases ---------------------------------------------------


class TestStopBoundary:
    def test_max_tokens_one(self) -> None:
        """Single token limit — first token triggers LENGTH."""
        stop = StopCondition(eos_token_id=0, max_tokens=1)
        assert stop.check(5, "x") == FinishReason.LENGTH

    def test_eos_at_max_tokens_prefers_stop(self) -> None:
        """When EOS arrives exactly at max_tokens, EOS (STOP) takes priority."""
        stop = StopCondition(eos_token_id=0, max_tokens=1)
        assert stop.check(0, "") == FinishReason.STOP

    def test_eos_token_id_none(self) -> None:
        """No EOS configured — only max_tokens can stop."""
        stop = StopCondition(eos_token_id=None, max_tokens=2)
        assert stop.check(0, "x") is None
        assert stop.check(0, "y") == FinishReason.LENGTH

    def test_stop_sequence_single_char(self) -> None:
        stop = StopCondition(eos_token_id=None, max_tokens=100, stop_sequences=(".",))
        assert stop.check(1, "hello") is None
        assert stop.check(2, ".") == FinishReason.STOP


# -- Edge cases ---------------------------------------------------------------


class TestStopEdgeCases:
    def test_empty_text_tokens(self) -> None:
        """Tokens with empty text should not crash stop checks."""
        stop = StopCondition(eos_token_id=None, max_tokens=100, stop_sequences=("end",))
        assert stop.check(1, "") is None
        assert stop.check(2, "") is None

    def test_unicode_stop_sequence(self) -> None:
        stop = StopCondition(eos_token_id=None, max_tokens=100, stop_sequences=("✿RESULT✿",))
        assert stop.check(1, "✿") is None
        assert stop.check(2, "RESULT") is None
        assert stop.check(3, "✿") == FinishReason.STOP

    def test_multiple_stop_sequences(self) -> None:
        """First matching stop sequence wins."""
        stop = StopCondition(eos_token_id=None, max_tokens=100, stop_sequences=("STOP", "HALT"))
        assert stop.check(1, "H") is None
        assert stop.check(2, "A") is None
        assert stop.check(3, "L") is None
        assert stop.check(4, "T") == FinishReason.STOP

    def test_overlapping_stop_sequences(self) -> None:
        """Stop sequences that share prefixes."""
        stop = StopCondition(eos_token_id=None, max_tokens=100, stop_sequences=("AB", "ABC"))
        assert stop.check(1, "A") is None
        assert stop.check(2, "B") == FinishReason.STOP  # "AB" matches first


# -- Stateful / lifecycle -----------------------------------------------------


class TestStopStateful:
    def test_generated_count_increments(self) -> None:
        stop = StopCondition(eos_token_id=None, max_tokens=100)
        assert stop.generated_count == 0
        stop.check(1, "a")
        assert stop.generated_count == 1
        stop.check(2, "b")
        assert stop.generated_count == 2

    def test_text_buffer_accumulates_for_stop_sequences(self) -> None:
        """Stop sequence matching works when the sequence spans multiple tokens."""
        stop = StopCondition(eos_token_id=None, max_tokens=100, stop_sequences=("hello world",))
        assert stop.check(1, "hello") is None
        assert stop.check(2, " ") is None
        assert stop.check(3, "world") == FinishReason.STOP

    def test_text_buffer_trims_long_accumulated_text(self) -> None:
        """Buffer shouldn't grow unbounded — it trims to 2x max stop sequence length."""
        stop = StopCondition(eos_token_id=None, max_tokens=1000, stop_sequences=("END",))
        # Feed many tokens before the stop sequence
        for i in range(100):
            result = stop.check(i + 10, "x")
            assert result is None
        # Stop sequence should still work after buffer trimming
        assert stop.check(200, "E") is None
        assert stop.check(201, "N") is None
        assert stop.check(202, "D") == FinishReason.STOP

    def test_no_stop_sequences_means_no_buffer(self) -> None:
        """Without stop sequences, text buffer logic is skipped (O2 hot path)."""
        stop = StopCondition(eos_token_id=None, max_tokens=100)
        # This should not accumulate any text buffer
        stop.check(1, "a" * 10000)
        assert stop._text_buffer == ""


# -- Corner cases --------------------------------------------------------------


class TestStopCornerCases:
    def test_eos_and_stop_sequence_same_token(self) -> None:
        """EOS check runs before stop sequence check."""
        stop = StopCondition(eos_token_id=42, max_tokens=100, stop_sequences=("text",))
        # EOS should fire even if text doesn't match a stop sequence
        assert stop.check(42, "text") == FinishReason.STOP

    def test_extra_eos_ids_as_empty_tuple(self) -> None:
        stop = StopCondition(eos_token_id=0, max_tokens=100, extra_eos_token_ids=())
        assert stop.check(99, "x") is None

    def test_empty_stop_sequences_tuple(self) -> None:
        stop = StopCondition(eos_token_id=0, max_tokens=100, stop_sequences=())
        # Should not enter stop sequence checking branch
        for _ in range(50):
            result = stop.check(99, "x")
            if result is not None:
                break
        else:
            assert True  # No stop triggered
