from __future__ import annotations

from mlxs.chat.present.commands import (
    COMMANDS,
    command_context,
    command_names,
    command_palette_items,
    help_card,
    render_command_palette_fragments,
)


def test_command_names_match_registered_commands() -> None:
    assert command_names() == tuple(f"/{spec.name}" for spec in COMMANDS)


def test_help_card_contains_expected_commands_and_shortcuts() -> None:
    text = help_card()
    assert "slash commands:" in text
    assert "/help" in text
    assert "/retry" in text
    assert "/              open command palette when composer is empty" in text
    assert "Enter          insert newline" in text
    assert "Ctrl+J         submit current input" in text
    assert "Ctrl+Up/Down   switch active conversation" in text
    assert "Ctrl+C" in text


def test_command_context_for_root_exact_and_unknown_matches() -> None:
    assert "press / from an empty composer to browse commands" in command_context("/")
    assert "/retry" in command_context("/retry")
    assert "unknown command" in command_context("/nope")


def test_command_palette_items_filter_and_insert_text() -> None:
    items = command_palette_items("hist")

    assert [item.name for item in items] == ["history"]
    assert items[0].insert_text == "/history "


def test_render_command_palette_fragments_marks_selected_item() -> None:
    items = command_palette_items("re")

    rendered = "".join(
        text for _, text in render_command_palette_fragments(items, selected_index=0, query="re")
    )

    assert "Commands" in rendered
    assert "/retry" in rendered
