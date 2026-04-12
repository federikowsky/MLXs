from __future__ import annotations

from mlxs.chat.tui.help_overlay import HelpOverlay


def test_help_overlay_renders_help_fragments() -> None:
    overlay = HelpOverlay(invalidate=lambda: None)
    overlay.open()

    rendered = "".join(text for _, text in overlay.fragments())

    assert "Help" in rendered
    assert "slash commands:" in rendered
    assert "/help" in rendered
