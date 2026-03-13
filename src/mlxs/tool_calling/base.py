"""Base tool call parser protocol (§6.6, FR7).

Defines the interface all tool call parsers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mlxs._types import ToolCallResult


class ToolCallParser(ABC):
    """Abstract base for tool call parsers.

    Each model family may produce tool calls in different formats
    (JSON, XML-like, model-specific tags). Parsers detect and extract
    structured tool calls from raw model output text.
    """

    @abstractmethod
    def parse(self, text: str) -> list[ToolCallResult]:
        """Parse tool calls from generated text.

        Args:
            text: Raw model output text.

        Returns:
            List of parsed tool calls. Empty if none detected.
        """

    @abstractmethod
    def is_tool_call(self, text: str) -> bool:
        """Quick check if text likely contains a tool call.

        Used for early detection without full parsing overhead.

        Args:
            text: Raw model output text (may be partial during streaming).

        Returns:
            True if text appears to contain a tool call.
        """
