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
from mlxs.chat.present.progress import build_progress_fragments, progress_visible
from mlxs.chat.present.sessions import (
    SessionListItem,
    empty_session_list_fragments,
    format_updated_label,
    item_from_summary,
    item_from_session,
    items_from_summaries,
    render_session_list_fragments,
)
from mlxs.chat.present.transcript import TranscriptEntry, empty_state_fragments, entry_from_message, render_entry

__all__ = [
    "COMMANDS",
    "CommandSpec",
    "SessionListItem",
    "TranscriptEntry",
    "build_composer_context_fragments",
    "build_composer_shortcuts_fragments",
    "build_footer_left_fragments",
    "build_footer_right_fragments",
    "build_header_fragments",
    "build_progress_fragments",
    "command_context",
    "command_names",
    "display_path",
    "empty_state_fragments",
    "empty_session_list_fragments",
    "entry_from_message",
    "format_updated_label",
    "help_card",
    "item_from_summary",
    "item_from_session",
    "items_from_summaries",
    "progress_visible",
    "render_session_list_fragments",
    "render_entry",
    "short_model_name",
    "truncate_text",
]
