"""Stop condition checking for generation (FR9, FR10).

Handles EOS tokens, max_tokens, stop sequences, and extra EOS token ids.
Designed for zero-allocation checking in the decode loop.
"""

from __future__ import annotations

from mlxs._types import FinishReason


class StopCondition:
    """Evaluates whether generation should stop.

    Constructed once before the decode loop. Check methods are designed
    to be called per-token with minimal overhead (O2).
    """

    __slots__ = (
        "_eos_token_id",
        "_extra_eos_ids",
        "_generated",
        "_has_stop_sequences",
        "_max_stop_sequence_len",
        "_max_tokens",
        "_stop_sequences",
        "_text_buffer",
        "_text_buffer_limit",
    )

    def __init__(
        self,
        *,
        eos_token_id: int | None,
        max_tokens: int,
        stop_sequences: tuple[str, ...] = (),
        extra_eos_token_ids: tuple[int, ...] = (),
    ) -> None:
        self._eos_token_id = eos_token_id
        self._extra_eos_ids = frozenset(extra_eos_token_ids)
        self._stop_sequences = stop_sequences
        self._max_tokens = max_tokens
        self._generated = 0
        self._has_stop_sequences = bool(stop_sequences)
        self._max_stop_sequence_len = max((len(s) for s in stop_sequences), default=0)
        self._text_buffer_limit = self._max_stop_sequence_len * 2
        self._text_buffer = ""

    @property
    def needs_text(self) -> bool:
        return self._has_stop_sequences

    @property
    def generated_count(self) -> int:
        return self._generated

    def check_token(self, token_id: int) -> FinishReason | None:
        """Check token-id and length based stopping."""
        self._generated += 1

        if token_id == self._eos_token_id or token_id in self._extra_eos_ids:
            return FinishReason.STOP

        if self._generated >= self._max_tokens:
            return FinishReason.LENGTH

        return None

    def check_text(self, text: str) -> FinishReason | None:
        """Check text-based stop sequences using a bounded rolling buffer."""
        if not self._has_stop_sequences:
            return None

        self._text_buffer += text
        if len(self._text_buffer) > self._text_buffer_limit:
            self._text_buffer = self._text_buffer[-self._max_stop_sequence_len :]

        for seq in self._stop_sequences:
            if seq in self._text_buffer:
                return FinishReason.STOP

        return None

    def check(self, token_id: int, text: str) -> FinishReason | None:
        """Check if generation should stop after this token.

        Args:
            token_id: The generated token id.
            text: The decoded text for this token.

        Returns:
            FinishReason if generation should stop, None otherwise.
        """
        finish = self.check_token(token_id)
        if finish is not None:
            return finish
        return self.check_text(text)
