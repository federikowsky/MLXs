"""Prompt-toolkit command palette overlay for the chat TUI."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput

from mlxs.chat.present.commands import (
    CommandPaletteItem,
    command_palette_items,
    render_command_palette_fragments,
)


class CommandPalette:
    """Owns command palette state and prompt_toolkit widgets."""

    __slots__ = (
        "_invalidate",
        "_selected_index",
        "_visible",
        "container",
        "filter_buffer",
        "filter_window",
        "list_window",
    )

    def __init__(self, *, invalidate: Callable[[], None]) -> None:
        self._invalidate = invalidate
        self._selected_index = 0
        self._visible = False
        self.filter_buffer = Buffer(multiline=False)
        self.filter_buffer.on_text_changed += self._on_filter_changed
        self.filter_window = Window(
            content=BufferControl(
                buffer=self.filter_buffer,
                input_processors=[BeforeInput([("class:palette.prompt", " / ")])],
            ),
            height=1,
            style="class:palette.filter",
            width=Dimension(min=44, preferred=56, max=68),
        )
        self.list_window = Window(
            content=FormattedTextControl(self.fragments),
            wrap_lines=True,
            height=Dimension(min=8, preferred=10, max=12),
            width=Dimension(min=44, preferred=56, max=68),
            style="class:palette",
            dont_extend_height=False,
        )
        self.container = ConditionalContainer(
            content=HSplit([self.filter_window, self.list_window]),
            filter=Condition(lambda: self._visible),
        )

    @property
    def visible(self) -> bool:
        return self._visible

    def open(self) -> None:
        self._visible = True
        self._selected_index = 0
        self.filter_buffer.document = Document("")
        self._invalidate()

    def close(self) -> None:
        self._visible = False
        self._selected_index = 0
        self.filter_buffer.document = Document("")
        self._invalidate()

    def move_selection(self, delta: int) -> None:
        items = self.items()
        if not items:
            return
        self._selected_index = (self._selected_index + delta) % len(items)
        self._invalidate()

    def items(self) -> list[CommandPaletteItem]:
        return command_palette_items(self.filter_buffer.text)

    def selected_item(self) -> CommandPaletteItem | None:
        items = self.items()
        if not items:
            return None
        index = min(max(self._selected_index, 0), len(items) - 1)
        return items[index]

    def fragments(self) -> list[tuple[str, str]]:
        return render_command_palette_fragments(
            self.items(),
            selected_index=self._selected_index,
            query=self.filter_buffer.text,
        )

    def _on_filter_changed(self, _buffer: Buffer) -> None:
        self._selected_index = 0
        self._invalidate()


__all__ = ["CommandPalette"]
