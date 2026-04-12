from __future__ import annotations

from mlxs.chat.tui.style import CHAT_STYLE


def test_chat_style_contains_core_shell_regions() -> None:
    style = CHAT_STYLE.style_rules
    selectors = {selector for selector, _ in style}

    assert "header.brand" in selectors
    assert "rail.title" in selectors
    assert "rail.item.active" in selectors
    assert "rail.filter" in selectors
    assert "rail.filter_prompt" in selectors
    assert "context.title" in selectors
    assert "context.label" in selectors
    assert "context.value" in selectors
    assert "transcript" in selectors
    assert "composer.prompt" in selectors
    assert "progress.state" in selectors
    assert "statusbar.state" in selectors
