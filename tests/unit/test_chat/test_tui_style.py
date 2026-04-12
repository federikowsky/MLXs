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
    assert "palette.title" in selectors
    assert "palette.section" in selectors
    assert "palette.item.active" in selectors
    assert "palette.filter" in selectors
    assert "picker.title" in selectors
    assert "picker.item.active" in selectors
    assert "picker.filter" in selectors
    assert "progress.state" in selectors
    assert "statusbar.state" in selectors
    assert "label.status" in selectors
    assert "label.help" in selectors
    assert "help.overlay.title" in selectors
    assert "help.overlay.body" in selectors
    assert "body.status" in selectors
    assert "body.help" in selectors
