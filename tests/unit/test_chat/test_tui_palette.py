from __future__ import annotations

from mlxs.chat.tui.palette import CommandPalette


def test_command_palette_opens_filters_and_selects_items() -> None:
    invalidate_calls: list[str] = []
    palette = CommandPalette(invalidate=lambda: invalidate_calls.append("invalidate"))

    palette.open()
    palette.filter_buffer.text = "ret"

    selected = palette.selected_item()
    rendered = "".join(text for _, text in palette.fragments())

    assert palette.visible is True
    assert selected is not None
    assert selected.name == "retry"
    assert "/retry" in rendered
    assert "Regenerate the last user turn." in rendered
    assert invalidate_calls


def test_command_palette_move_selection_wraps() -> None:
    palette = CommandPalette(invalidate=lambda: None)
    palette.open()

    first = palette.selected_item()
    palette.move_selection(1)
    second = palette.selected_item()
    palette.move_selection(-1)

    assert first is not None
    assert second is not None
    assert first.name != second.name
    assert palette.selected_item() == first
