"""Tool calling — parse tool calls from model output (§6.6, FR7, AC15).

Dispatches to model-family-specific parsers. Fully isolated module —
depends only on _types.py.
"""

from __future__ import annotations

from mlxs._types import ToolCallResult
from mlxs.tool_calling.base import ToolCallParser
from mlxs.tool_calling.generic import GenericToolCallParser
from mlxs.tool_calling.qwen import QwenToolCallParser

# Parser registry: parser name → class
_PARSERS: dict[str, type[ToolCallParser]] = {
    "generic": GenericToolCallParser,
    "qwen": QwenToolCallParser,
}


def get_parser(parser_name: str) -> ToolCallParser:
    """Get a tool call parser by name (§6.6).

    Args:
        parser_name: Parser identifier from config (tool_call_parser).

    Returns:
        Configured parser instance.

    Raises:
        ValueError: If parser_name is not registered.
    """
    cls = _PARSERS.get(parser_name)
    if cls is None:
        supported = sorted(_PARSERS)
        raise ValueError(
            f"Unknown tool_call_parser '{parser_name}'. Supported: {', '.join(supported)}"
        )
    return cls()


__all__ = [
    "ToolCallParser",
    "ToolCallResult",
    "get_parser",
]
