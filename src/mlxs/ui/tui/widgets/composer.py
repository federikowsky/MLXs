"""Composer widget — multiline input area for chat messages.

Supports:
- Enter to send (when not shifted)
- Shift+Enter for newline
- Slash commands (lines starting with /)
- Placeholder text
"""

from __future__ import annotations

from textual.message import Message
from textual.widgets import TextArea


class Composer(TextArea):
    """Multiline input area for composing chat messages."""

    DEFAULT_CSS = """
    Composer {
        height: auto;
        max-height: 8;
        min-height: 3;
        dock: bottom;
        border: tall $primary;
        padding: 0 1;
    }
    Composer:focus {
        border: tall $accent;
    }
    """

    class Submitted(Message):
        """Posted when the user submits a message (Enter without Shift)."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs) -> None:
        super().__init__(
            language=None,
            theme="monokai",
            soft_wrap=True,
            show_line_numbers=False,
            **kwargs,
        )

    def on_mount(self) -> None:
        self.focus()

    async def _on_key(self, event) -> None:
        """Handle Enter vs Shift+Enter."""
        if event.key == "enter" and not event.shift_key:
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
            return
        # Let Shift+Enter produce a newline (default TextArea behaviour)
        await super()._on_key(event)
