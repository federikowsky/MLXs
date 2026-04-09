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
        "_max_stop_sequence_length",
        "_max_tokens",
        "_stop_sequences",
        "_text_buffer",
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
        self._max_stop_sequence_length = max((len(s) for s in stop_sequences), default=0)
        self._generated = 0
        self._text_buffer = ""

    def check(self, token_id: int, text: str) -> FinishReason | None:
        """Check if generation should stop after this token.

        Args:
            token_id: The generated token id.
            text: The decoded text for this token.

        Returns:
            FinishReason if generation should stop, None otherwise.
        """
        self._generated += 1

        # EOS token check
        if token_id == self._eos_token_id or token_id in self._extra_eos_ids:
            return FinishReason.STOP

        # Max tokens check
        if self._generated >= self._max_tokens:
            return FinishReason.LENGTH

        # Stop sequence check
        if self._stop_sequences:
            self._text_buffer += text
            # Only keep enough buffer for the longest stop sequence
            if len(self._text_buffer) > self._max_stop_sequence_length * 2:
                self._text_buffer = self._text_buffer[-self._max_stop_sequence_length :]
            for seq in self._stop_sequences:
                if seq in self._text_buffer:
                    return FinishReason.STOP

        return None

    def will_stop_at_next(self, generated_count: int) -> bool:
        """Return True when the next token must stop due to length."""
        return generated_count + 1 >= self._max_tokens

    @property
    def generated_count(self) -> int:
        return self._generated
