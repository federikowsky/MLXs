"""Transcript widget — displays the conversation as styled message bubbles.

Uses RichLog for efficient append-only rendering with Rich markup
and Markdown formatting for assistant messages.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.widgets import RichLog


class Transcript(RichLog):
    """Scrollable transcript of user/assistant messages."""

    DEFAULT_CSS = """
    Transcript {
        height: 1fr;
        border: none;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            highlight=False,
            markup=True,
            wrap=True,
            auto_scroll=True,
            **kwargs,
        )
        self._streaming_text: list[str] = []

    def add_user_message(self, content: str) -> None:
        """Append a styled user message to the transcript."""
        user_text = Text(content, style="bold cyan")
        panel = Panel(
            user_text,
            title="[bold cyan]You[/]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
        self.write(panel)

    def start_assistant_message(self) -> None:
        """Begin a new assistant streaming message."""
        self._streaming_text.clear()
        self.write(Text(""))  # separator line

    def append_token(self, text: str) -> None:
        """Append a token to the current streaming assistant message.

        Uses plain text append for performance during streaming.
        """
        self._streaming_text.append(text)
        # Write individual tokens as plain text during streaming
        self.write(Text(text, style="white", end=""), scroll_end=False)

    def finish_assistant_message(self, full_text: str) -> None:
        """Replace in-progress streaming with a final formatted message.

        Clears the raw streaming tokens and renders the full response
        as a nicely formatted Panel with Markdown.
        """
        self._streaming_text.clear()
        if not full_text.strip():
            return
        md = Markdown(full_text)
        panel = Panel(
            md,
            title="[bold green]Assistant[/]",
            title_align="left",
            border_style="green",
            padding=(0, 1),
        )
        self.write(panel)

    def add_system_message(self, content: str) -> None:
        """Add a system/status message."""
        self.write(Text(f"  {content}", style="dim italic"))

    def add_error_message(self, content: str) -> None:
        """Add an error message."""
        self.write(Text(f"  Error: {content}", style="bold red"))

    def clear_transcript(self) -> None:
        """Clear all content."""
        self.clear()
