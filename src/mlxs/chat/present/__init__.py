"""Presentation-layer helpers for chat product surfaces."""

from mlxs.chat.present.context import (
    ContextRailRow,
    ContextRailSummary,
    build_context_rail_summary,
    empty_context_rail_fragments,
    render_context_rail_fragments,
)
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
from mlxs.chat.present.confirmation import ConfirmationSpec, render_confirmation_fragments
from mlxs.chat.present.commands import (
    COMMANDS,
    CommandPaletteItem,
    CommandSpec,
    command_context,
    command_names,
    command_palette_items,
    help_card,
    render_help_overlay_fragments,
    render_command_palette_fragments,
)
from mlxs.chat.present.progress import build_progress_fragments, progress_visible
from mlxs.chat.present.references import ReferencePickerItem, render_reference_picker_fragments
from mlxs.chat.present.sessions import (
    SessionListItem,
    empty_session_list_fragments,
    filter_session_items,
    format_updated_label,
    item_from_summary,
    item_from_session,
    items_from_summaries,
    render_session_list_fragments,
)
from mlxs.chat.present.transcript import TranscriptEntry, empty_state_fragments, entry_from_message, render_entry

__all__ = [
    "COMMANDS",
    "CommandPaletteItem",
    "CommandSpec",
    "ConfirmationSpec",
    "ContextRailRow",
    "ContextRailSummary",
    "ReferencePickerItem",
    "SessionListItem",
    "TranscriptEntry",
    "build_context_rail_summary",
    "build_composer_context_fragments",
    "build_composer_shortcuts_fragments",
    "build_footer_left_fragments",
    "build_footer_right_fragments",
    "build_header_fragments",
    "build_progress_fragments",
    "command_context",
    "command_names",
    "command_palette_items",
    "display_path",
    "empty_context_rail_fragments",
    "empty_state_fragments",
    "empty_session_list_fragments",
    "entry_from_message",
    "filter_session_items",
    "format_updated_label",
    "help_card",
    "item_from_summary",
    "item_from_session",
    "items_from_summaries",
    "progress_visible",
    "render_help_overlay_fragments",
    "render_command_palette_fragments",
    "render_confirmation_fragments",
    "render_context_rail_fragments",
    "render_reference_picker_fragments",
    "render_session_list_fragments",
    "render_entry",
    "short_model_name",
    "truncate_text",
]
