from __future__ import annotations

from mlxs.chat.present.commands import COMMANDS, command_context, command_names, help_card


def test_command_names_match_registered_commands() -> None:
    assert command_names() == tuple(f"/{spec.name}" for spec in COMMANDS)


def test_help_card_contains_expected_commands_and_shortcuts() -> None:
    text = help_card()
    assert "slash commands:" in text
    assert "/help" in text
    assert "/retry" in text
    assert "Ctrl+C" in text


def test_command_context_for_root_exact_and_unknown_matches() -> None:
    assert "Tab shows the full command menu" in command_context("/")
    assert "/retry" in command_context("/retry")
    assert "unknown command" in command_context("/nope")
