"""Layer 2 stop-sequence handling above the minimal Layer 1 core finish."""

from __future__ import annotations


class StopSequenceMatcher:
    """Track textual stop-sequence matches for one generation flow."""

    __slots__ = ("_stop_sequences", "_text_buffer")

    def __init__(self, stop_sequences: tuple[str, ...] = ()) -> None:
        self._stop_sequences = stop_sequences
        self._text_buffer = ""

    def check(self, text: str) -> bool:
        """Return True when a configured stop sequence has been observed."""
        if not self._stop_sequences:
            return False
        self._text_buffer += text
        max_len = max(len(seq) for seq in self._stop_sequences)
        if len(self._text_buffer) > max_len * 2:
            self._text_buffer = self._text_buffer[-max_len:]
        return any(seq in self._text_buffer for seq in self._stop_sequences)
