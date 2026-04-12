"""Prompt-toolkit confirmation dialog overlay for the chat TUI."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from mlxs.chat.present.confirmation import ConfirmationSpec, render_confirmation_fragments


class ConfirmationDialog:
    """Owns the passive confirmation dialog state and widget."""

    __slots__ = ("_invalidate", "_spec", "_visible", "container", "window")

    def __init__(self, *, invalidate: Callable[[], None]) -> None:
        self._invalidate = invalidate
        self._spec: ConfirmationSpec | None = None
        self._visible = False
        self.window = Window(
            content=FormattedTextControl(self.fragments, focusable=True),
            wrap_lines=True,
            height=Dimension(min=6, preferred=8, max=10),
            width=Dimension(min=40, preferred=56, max=64),
            style="class:confirm",
            dont_extend_height=False,
        )
        self.container = ConditionalContainer(
            content=HSplit([self.window]),
            filter=Condition(lambda: self._visible),
        )

    @property
    def visible(self) -> bool:
        return self._visible

    def open(self, spec: ConfirmationSpec) -> None:
        self._spec = spec
        self._visible = True
        self._invalidate()

    def close(self) -> None:
        self._visible = False
        self._spec = None
        self._invalidate()

    def fragments(self) -> list[tuple[str, str]]:
        return render_confirmation_fragments(self._spec)


__all__ = ["ConfirmationDialog"]
