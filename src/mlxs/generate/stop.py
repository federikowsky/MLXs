"""Stop matching for generation.

The decode engine keeps stop handling host-side to preserve exact text-buffer
semantics while keeping the device step minimal.
"""

from __future__ import annotations

from mlxs._types import FinishReason


class StopMatcher:
    """Evaluate EOS, max-token, and stop-sequence termination."""

    __slots__ = (
        "_eos_token_id",
        "_extra_eos_ids",
        "_generated",
        "_max_stop_sequence_len",
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
        self._generated = 0
        self._max_tokens = max_tokens
        self._stop_sequences = stop_sequences
        self._text_buffer = ""
        self._max_stop_sequence_len = max((len(seq) for seq in stop_sequences), default=0)

    def check(self, token_id: int, text: str) -> FinishReason | None:
        self._generated += 1

        if token_id == self._eos_token_id or token_id in self._extra_eos_ids:
            return FinishReason.STOP

        if self._generated >= self._max_tokens:
            return FinishReason.LENGTH

        if not self._stop_sequences:
            return None

        self._text_buffer += text
        if (
            self._max_stop_sequence_len > 0
            and len(self._text_buffer) > self._max_stop_sequence_len * 2
        ):
            self._text_buffer = self._text_buffer[-self._max_stop_sequence_len :]
        for seq in self._stop_sequences:
            if seq in self._text_buffer:
                return FinishReason.STOP
        return None

    @property
    def generated_count(self) -> int:
        return self._generated


# Preserve the existing import surface for legacy call sites and tests.
StopCondition = StopMatcher
