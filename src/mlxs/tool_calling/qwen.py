"""Qwen-style tool call parser (§6.6, FR7).

Qwen models emit tool calls using special tokens and a specific format:
✿FUNCTION✿: function_name
✿ARGS✿: {"arg": "value"}
✿RESULT✿:

or the newer format with <tool_call> tags.
"""

from __future__ import annotations

import json
import re
import uuid

from mlxs._types import ToolCallResult
from mlxs.tool_calling.base import ToolCallParser

# Qwen legacy format: ✿FUNCTION✿ / ✿ARGS✿
_QWEN_LEGACY_RE = re.compile(
    r"✿FUNCTION✿:\s*(.+?)\n✿ARGS✿:\s*(\{.*?\})(?:\n✿RESULT✿)?",
    re.DOTALL,
)

# Qwen2/3 format: <tool_call> JSON </tool_call>
_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


class QwenToolCallParser(ToolCallParser):
    """Parse Qwen-style tool calls.

    Supports both legacy (✿FUNCTION✿/✿ARGS✿) and modern
    (<tool_call>JSON</tool_call>) formats.
    """

    def parse(self, text: str) -> list[ToolCallResult]:
        """Parse tool calls from Qwen model output."""
        # Try modern format first
        results = self._parse_tool_call_tags(text)
        if results:
            return results

        # Fall back to legacy format
        return self._parse_legacy(text)

    def is_tool_call(self, text: str) -> bool:
        """Quick check for Qwen tool call markers."""
        return "<tool_call>" in text or "✿FUNCTION✿" in text

    def _parse_tool_call_tags(self, text: str) -> list[ToolCallResult]:
        """Parse <tool_call>JSON</tool_call> format."""
        results: list[ToolCallResult] = []
        for match in _QWEN_TOOL_CALL_RE.finditer(text):
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

            name = data.get("name")
            args = data.get("arguments", {})
            if name:
                results.append(
                    ToolCallResult(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=name,
                        arguments=json.dumps(args) if isinstance(args, dict) else str(args),
                    )
                )
        return results

    def _parse_legacy(self, text: str) -> list[ToolCallResult]:
        """Parse ✿FUNCTION✿/✿ARGS✿ format."""
        results: list[ToolCallResult] = []
        for match in _QWEN_LEGACY_RE.finditer(text):
            name = match.group(1).strip()
            args_str = match.group(2).strip()
            try:
                json.loads(args_str)
            except json.JSONDecodeError:
                continue
            results.append(
                ToolCallResult(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=name,
                    arguments=args_str,
                )
            )
        return results
