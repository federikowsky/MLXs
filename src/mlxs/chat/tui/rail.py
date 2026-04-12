"""Prompt-toolkit rail helpers for the chat TUI."""

from __future__ import annotations

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput

from mlxs.chat.present.sessions import SessionListItem, filter_session_items, render_session_list_fragments


class SessionRail:
    """Owns the passive session rail view state and prompt_toolkit widgets."""

    __slots__ = ("_items", "container", "filter_buffer", "filter_window", "list_window")

    def __init__(self) -> None:
        self._items: list[SessionListItem] = []
        self.filter_buffer = Buffer(multiline=False)
        self.filter_window = Window(
            content=BufferControl(
                buffer=self.filter_buffer,
                input_processors=[BeforeInput([("class:rail.filter_prompt", " Filter: ")])],
            ),
            height=1,
            style="class:rail.filter",
        )
        self.list_window = Window(
            content=FormattedTextControl(self.fragments),
            wrap_lines=True,
            style="class:rail",
            width=Dimension(min=22, preferred=28, max=32),
            dont_extend_height=False,
        )
        self.container = HSplit(
            [
                self.filter_window,
                self.list_window,
            ]
        )

    def set_items(self, items: list[SessionListItem]) -> None:
        self._items = list(items)

    def fragments(self) -> list[tuple[str, str]]:
        return render_session_list_fragments(
            filter_session_items(self._items, query=self.filter_buffer.text)
        )


def build_session_rail_window(
    fragments_fn,
):
    """Build the passive session rail window.

    The caller supplies the current fragment-producing callback so the rail stays
    a pure prompt_toolkit concern without owning product state.
    """
    return Window(
        content=FormattedTextControl(fragments_fn),
        wrap_lines=True,
        style="class:rail",
        width=Dimension(min=22, preferred=28, max=32),
        dont_extend_height=False,
    )


def session_rail_fragments(items: list[SessionListItem]) -> list[tuple[str, str]]:
    """Render rail fragments from the current session items."""
    return render_session_list_fragments(items)


__all__ = ["SessionRail", "build_session_rail_window", "session_rail_fragments"]
