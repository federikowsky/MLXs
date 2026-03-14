"""Status bar widget — displays model info, generation state, session details.

Docked to the bottom, above the composer.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Minimal status bar showing model, temperature, tokens, and generation state."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    model_name: reactive[str] = reactive("No model")
    temperature: reactive[float] = reactive(1.0)
    max_tokens: reactive[int] = reactive(512)
    status_text: reactive[str] = reactive("Ready")
    session_title: reactive[str] = reactive("New chat")

    def render(self) -> str:
        parts = [
            f"[bold]{self.model_name}[/]",
            f"T={self.temperature:.1f}",
            f"max={self.max_tokens}",
            f"[italic]{self.session_title}[/]",
            f"[{self._status_color}]{self.status_text}[/]",
        ]
        return "  │  ".join(parts)

    @property
    def _status_color(self) -> str:
        if "generating" in self.status_text.lower():
            return "bold yellow"
        if "error" in self.status_text.lower():
            return "bold red"
        return "green"
