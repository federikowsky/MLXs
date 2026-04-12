"""Chat template utilities for prompt building, stop-token discovery, output sanitization.

Centralises chat template logic so both the CLI chat and the HTTP route
use the same, correct pipeline.  Two entry points:

- ``build_prompt_ids``: tokenize=True path for the CLI chat -- no intermediate string.
- ``build_prompt_str``: tokenize=False path (HTTP route, needed for media embedding).

Plus helpers:
- ``collect_stop_token_ids``: discover model-specific termination tokens.
- ``sanitize_assistant_text``: strip leaked special markers from decoded text.
- ``StreamingTextSanitizer``: strip leaked markers while streaming chunk-by-chunk.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Common chat-model termination markers that may not be the primary eos_token.
_KNOWN_STOP_MARKERS: tuple[str, ...] = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|endoftext|>",
    "<|end|>",
    "</s>",
)

# Regex to strip any special markers from decoded assistant text.
_MARKER_RE = re.compile(r"<\|[^|>]+\|>")


def build_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Build prompt token ids from messages using tokenize=True (no intermediate string).

    This is the correct pipeline for the CLI chat: apply_chat_template returns
    token ids directly, avoiding the double-format bug.

    Falls back to a structured prompt if apply_chat_template is unavailable.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(result, list) and result:
                return list(result)
        except Exception as exc:
            logger.warning("apply_chat_template(tokenize=True) failed: %s; using fallback", exc)

    # Fallback: structured prompt
    return _build_fallback_ids(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
    )


def build_prompt_str(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Build prompt string from messages using tokenize=False.

    Used by the HTTP route where a string prompt is needed for media/embedding
    alignment.  Falls back to structured concatenation.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            result = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            if isinstance(result, str) and result:
                return result
        except Exception as exc:
            logger.warning("apply_chat_template(tokenize=False) failed: %s; using fallback", exc)

    return _build_fallback_str(messages)


def collect_stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Discover additional stop/EOS token ids from the tokenizer.

    Tries to resolve known special marker strings to token ids.
    Returns only ids that differ from the primary eos_token_id.
    """
    primary_eos = getattr(tokenizer, "eos_token_id", None)
    found: set[int] = set()

    inner = getattr(tokenizer, "inner", tokenizer)
    # Try added_tokens_encoder (HuggingFace tokenizers)
    added_tokens = getattr(inner, "added_tokens_encoder", {})

    for marker in _KNOWN_STOP_MARKERS:
        # Check added_tokens_encoder first (fast)
        if marker in added_tokens:
            tid = added_tokens[marker]
            if tid != primary_eos:
                found.add(tid)
            continue

        # Try convert_tokens_to_ids
        convert_fn = getattr(inner, "convert_tokens_to_ids", None)
        if convert_fn is not None:
            try:
                tid = convert_fn(marker)
                unk_id = getattr(inner, "unk_token_id", None)
                if tid is not None and tid != unk_id and tid != primary_eos:
                    found.add(tid)
            except Exception:
                pass

    if found:
        logger.info("Discovered extra stop token ids: %s", found)
    return tuple(sorted(found))


def sanitize_assistant_text(text: str) -> str:
    """Strip special markers from decoded assistant text.

    Removes known chat-template markers that leak into the decoded output
    (e.g. im_end, eot_id).  Returns cleaned text.
    """
    cleaned = _MARKER_RE.sub("", text)
    # Also strip any trailing whitespace that was before the marker
    return cleaned.rstrip()


class StreamingTextSanitizer:
    """Sanitize streamed chunks without leaking partial special markers."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        """Return the safe text that can be printed immediately."""
        if not text:
            return ""
        self._pending += text
        hold_back = _pending_marker_prefix_len(self._pending)
        emit_upto = len(self._pending) - hold_back
        if emit_upto <= 0:
            return ""
        emit = self._pending[:emit_upto]
        self._pending = self._pending[emit_upto:]
        return _MARKER_RE.sub("", emit)

    def flush(self) -> str:
        """Return any remaining safe text after the stream ends."""
        if not self._pending:
            return ""
        tail = sanitize_assistant_text(self._pending)
        self._pending = ""
        return tail


def _build_fallback_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Structured fallback: build prompt string then tokenize."""
    prompt_str = _build_fallback_str(
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    return list(tokenizer.encode(prompt_str))


def _build_fallback_str(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Structured fallback prompt: User:/Assistant: format."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n".join(parts)


def _pending_marker_prefix_len(text: str) -> int:
    """Longest suffix of ``text`` that may be the start of a stop marker."""
    best = 0
    for marker in _KNOWN_STOP_MARKERS:
        max_prefix = min(len(text), len(marker) - 1)
        for size in range(max_prefix, 0, -1):
            if text.endswith(marker[:size]):
                if size > best:
                    best = size
                break
    return best
