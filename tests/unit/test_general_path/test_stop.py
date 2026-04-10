"""Tests for Layer 2 stop-sequence handling."""

from __future__ import annotations

from mlxs.general_path.stop import StopSequenceMatcher


def test_stop_sequence_matcher_no_sequences() -> None:
    matcher = StopSequenceMatcher()
    assert matcher.check("hello") is False


def test_stop_sequence_matcher_detects_sequence_across_tokens() -> None:
    matcher = StopSequenceMatcher(("END",))
    assert matcher.check("E") is False
    assert matcher.check("N") is False
    assert matcher.check("D") is True


def test_stop_sequence_matcher_trims_buffer() -> None:
    matcher = StopSequenceMatcher(("STOP",))
    for _ in range(20):
        assert matcher.check("x") is False
    assert matcher.check("S") is False
    assert matcher.check("TO") is False
    assert matcher.check("P") is True
