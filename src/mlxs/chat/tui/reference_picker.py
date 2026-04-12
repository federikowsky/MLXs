"""Prompt-toolkit @ reference picker overlay for the chat TUI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput

from mlxs.chat.present.references import ReferencePickerItem, render_reference_picker_fragments
from mlxs.chat.tui.completion import reference_candidate_values


class ReferencePicker:
    """Owns @ picker state and prompt_toolkit widgets."""

    __slots__ = (
        "_cwd",
        "_invalidate",
        "_replace_end",
        "_replace_start",
        "_selected_index",
        "_visible",
        "container",
        "filter_buffer",
        "filter_window",
        "list_window",
    )

    def __init__(self, *, cwd: Callable[[], Path], invalidate: Callable[[], None]) -> None:
        self._cwd = cwd
        self._invalidate = invalidate
        self._replace_start = 0
        self._replace_end = 0
        self._selected_index = 0
        self._visible = False
        self.filter_buffer = Buffer(multiline=False)
        self.filter_buffer.on_text_changed += self._on_filter_changed
        self.filter_window = Window(
            content=BufferControl(
                buffer=self.filter_buffer,
                input_processors=[BeforeInput([("class:picker.prompt", " @ ")])],
            ),
            height=1,
            style="class:picker.filter",
            width=Dimension(min=44, preferred=56, max=68),
        )
        self.list_window = Window(
            content=FormattedTextControl(self.fragments),
            wrap_lines=True,
            height=Dimension(min=8, preferred=10, max=12),
            width=Dimension(min=44, preferred=56, max=68),
            style="class:picker",
            dont_extend_height=False,
        )
        self.container = ConditionalContainer(
            content=HSplit([self.filter_window, self.list_window]),
            filter=Condition(lambda: self._visible),
        )

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def replace_start(self) -> int:
        return self._replace_start

    @property
    def replace_end(self) -> int:
        return self._replace_end

    def open(self, *, replace_start: int, replace_end: int, initial_query: str = "") -> None:
        self._replace_start = replace_start
        self._replace_end = replace_end
        self._visible = True
        self._selected_index = 0
        self.filter_buffer.document = Document(initial_query, cursor_position=len(initial_query))
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

    def items(self) -> list[ReferencePickerItem]:
        return [
            ReferencePickerItem(label=value, insert_text=f"@{value}", meta="attach file")
            for value in reference_candidate_values(self.filter_buffer.text, self._cwd())
        ]

    def selected_item(self) -> ReferencePickerItem | None:
        items = self.items()
        if not items:
            return None
        index = min(max(self._selected_index, 0), len(items) - 1)
        return items[index]

    def fragments(self) -> list[tuple[str, str]]:
        return render_reference_picker_fragments(
            self.items(),
            selected_index=self._selected_index,
            query=self.filter_buffer.text,
        )

    def _on_filter_changed(self, _buffer: Buffer) -> None:
        self._selected_index = 0
        self._invalidate()


__all__ = ["ReferencePicker"]
