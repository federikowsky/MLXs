"""Presentation-layer helpers for chat product surfaces."""

from mlxs.chat.present.commands import COMMANDS, CommandSpec, command_context, command_names, help_card
from mlxs.chat.present.transcript import TranscriptEntry, empty_state_fragments, entry_from_message, render_entry

__all__ = [
    "COMMANDS",
    "CommandSpec",
    "TranscriptEntry",
    "command_context",
    "command_names",
    "empty_state_fragments",
    "entry_from_message",
    "help_card",
    "render_entry",
]
