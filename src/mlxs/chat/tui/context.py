"""Prompt-toolkit helpers for the passive right context rail."""

from __future__ import annotations

from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from mlxs.chat.present.context import (
    ContextRailSummary,
    render_context_rail_fragments,
)


class ContextRail:
    """Owns the passive right context rail summary and window."""

    __slots__ = ("_summary", "window")

    def __init__(self) -> None:
        self._summary: ContextRailSummary | None = None
        self.window = Window(
            content=FormattedTextControl(self.fragments),
            wrap_lines=True,
            style="class:context",
            width=Dimension(min=20, preferred=24, max=28),
            dont_extend_height=False,
        )

    def set_summary(self, summary: ContextRailSummary) -> None:
        self._summary = summary

    def fragments(self) -> list[tuple[str, str]]:
        return render_context_rail_fragments(self._summary)


__all__ = ["ContextRail"]
