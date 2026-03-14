"""Tokenizer loading — protocol-compatible wrapper (FR2, §7.2).

Wraps any tokenizer-like object (HuggingFace, mlx_lm, etc.) with the minimal
interface needed by generate: encode, decode, eos_token_id, vocab_size,
and optionally apply_chat_template (with fallback when missing).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _fallback_chat_template(messages: list[dict[str, Any]]) -> str:
    """Concatenate message contents when no chat template is available."""
    return "\n".join(m.get("content", "") for m in messages)


class TokenizerWrapper:
    """Thin wrapper around any tokenizer satisfying TokenizerProtocol.

    Accepts HuggingFace PreTrainedTokenizerBase or any object with encode,
    decode, eos_token_id, vocab_size; optionally apply_chat_template.
    """

    __slots__ = ("_tokenizer",)

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (with special tokens)."""
        return list(self._tokenizer.encode(text))

    def decode(self, token_ids: list[int] | int) -> str:
        """Decode token ids to text."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self._tokenizer.decode(token_ids)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self._tokenizer, "eos_token_id", None)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def inner(self) -> Any:
        """Access the underlying tokenizer for advanced operations."""
        return self._tokenizer

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str | list[int]:
        """Apply chat template if available; otherwise fallback to concatenating contents."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        return _fallback_chat_template(messages)


def load_hf_tokenizer(
    model_path: str | Path,
    *,
    trust_remote_code: bool = False,
) -> TokenizerWrapper:
    """Load a HuggingFace tokenizer and wrap it.

    Args:
        model_path: Local path or HF repo id.
        trust_remote_code: Allow custom tokenizer code.

    Returns:
        TokenizerWrapper instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
    )
    logger.info("Loaded tokenizer from %s (vocab_size=%d)", model_path, tokenizer.vocab_size)
    return TokenizerWrapper(tokenizer)
