"""Prompt-toolkit help overlay for the chat TUI."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from mlxs.chat.present.commands import render_help_overlay_fragments


class HelpOverlay:
    """Owns the passive help overlay state and widget."""

    __slots__ = ("_invalidate", "_visible", "container", "window")

    def __init__(self, *, invalidate: Callable[[], None]) -> None:
        self._invalidate = invalidate
        self._visible = False
        self.window = Window(
            content=FormattedTextControl(self.fragments, focusable=True),
            wrap_lines=True,
            height=Dimension(min=12, preferred=16, max=20),
            width=Dimension(min=48, preferred=68, max=84),
            style="class:help.overlay",
            dont_extend_height=False,
        )
        self.container = ConditionalContainer(
            content=HSplit(
                [
                    self.window
                ]
            ),
            filter=Condition(lambda: self._visible),
        )

    @property
    def visible(self) -> bool:
        return self._visible

    def open(self) -> None:
        self._visible = True
        self._invalidate()

    def close(self) -> None:
        self._visible = False
        self._invalidate()

    def fragments(self) -> list[tuple[str, str]]:
        return render_help_overlay_fragments()


__all__ = ["HelpOverlay"]
