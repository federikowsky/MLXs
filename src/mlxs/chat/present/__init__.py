"""Presentation-layer helpers for chat product surfaces."""

from mlxs.chat.present.chrome import (
    build_composer_context_fragments,
    build_composer_shortcuts_fragments,
    build_footer_left_fragments,
    build_footer_right_fragments,
    build_header_fragments,
    display_path,
    short_model_name,
    truncate_text,
)
from mlxs.chat.present.commands import COMMANDS, CommandSpec, command_context, command_names, help_card
from mlxs.chat.present.transcript import TranscriptEntry, empty_state_fragments, entry_from_message, render_entry

__all__ = [
    "COMMANDS",
    "CommandSpec",
    "TranscriptEntry",
    "build_composer_context_fragments",
    "build_composer_shortcuts_fragments",
    "build_footer_left_fragments",
    "build_footer_right_fragments",
    "build_header_fragments",
    "command_context",
    "command_names",
    "display_path",
    "empty_state_fragments",
    "entry_from_message",
    "help_card",
    "render_entry",
    "short_model_name",
    "truncate_text",
]
