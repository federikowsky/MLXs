"""Generic / OpenAI-style tool call parser (§6.6, FR7).

Parses tool calls from JSON function call format commonly used by
OpenAI-compatible models. Expects output containing JSON objects with
"name" and "arguments" fields.
"""

from __future__ import annotations

import json
import re
import uuid

from mlxs._types import ToolCallResult
from mlxs.tool_calling.base import ToolCallParser

# Match JSON objects that look like function calls
_FUNCTION_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
    re.DOTALL,
)

# Alternative: tool_calls array format
_TOOL_CALLS_RE = re.compile(
    r'"tool_calls"\s*:\s*\[([^\]]*)\]',
    re.DOTALL,
)


class GenericToolCallParser(ToolCallParser):
    """Parse OpenAI-style function/tool calls from model output.

    Handles two common formats:
    1. Inline JSON: {"name": "func", "arguments": {"arg": "val"}}
    2. tool_calls array: [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
    """

    def parse(self, text: str) -> list[ToolCallResult]:
        """Parse tool calls from text."""
        # Try tool_calls array format first
        results = self._parse_tool_calls_array(text)
        if results:
            return results

        # Fall back to inline function call format
        return self._parse_inline_calls(text)

    def is_tool_call(self, text: str) -> bool:
        """Quick check for tool call markers."""
        return '"name"' in text and '"arguments"' in text

    def _parse_inline_calls(self, text: str) -> list[ToolCallResult]:
        """Parse inline {"name": ..., "arguments": ...} patterns."""
        results: list[ToolCallResult] = []
        for match in _FUNCTION_CALL_RE.finditer(text):
            name = match.group(1)
            args_str = match.group(2)
            # Validate arguments is valid JSON
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

    def _parse_tool_calls_array(self, text: str) -> list[ToolCallResult]:
        """Parse tool_calls array format."""
        match = _TOOL_CALLS_RE.search(text)
        if not match:
            return []

        try:
            # Re-parse the full tool_calls structure
            # Find the enclosing JSON object
            start = text.rfind("{", 0, match.start())
            if start == -1:
                return []
            data = json.loads(text[start:])
            tool_calls = data.get("tool_calls", [])
        except (json.JSONDecodeError, KeyError):
            return []

        results: list[ToolCallResult] = []
        for call in tool_calls:
            func = call.get("function", call)
            name = func.get("name")
            args = func.get("arguments", "{}")
            call_id = call.get("id", f"call_{uuid.uuid4().hex[:12]}")
            if name:
                if isinstance(args, dict):
                    args = json.dumps(args)
                results.append(ToolCallResult(id=call_id, name=name, arguments=args))
        return results
