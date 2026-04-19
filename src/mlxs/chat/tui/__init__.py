"""Prompt-toolkit UI helpers for chat product surfaces."""

from mlxs.chat.tui.completion import ChatCompleter, build_completions
try:
    from mlxs.chat.tui.confirmation import ConfirmationDialog
    from mlxs.chat.tui.context import ContextRail
    from mlxs.chat.tui.help_overlay import HelpOverlay
    from mlxs.chat.tui.keymap import build_chat_key_bindings
    from mlxs.chat.tui.overlay import OverlaySpec, anchored_overlay, build_overlay_host, positioned_overlay
    from mlxs.chat.tui.palette import CommandPalette
    from mlxs.chat.tui.reference_picker import ReferencePicker
    from mlxs.chat.tui.rail import SessionRail, build_session_rail_window, session_rail_fragments
    from mlxs.chat.tui.scaffold import (
        BodyScaffoldParts,
        ShellScaffoldParts,
        build_body_scaffold,
        build_transcript_first_scaffold,
    )
    from mlxs.chat.tui.shell import ChatShell, RepoContext, discover_repo_context
    from mlxs.chat.tui.style import CHAT_STYLE
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via cli import tests
    if exc.name != "prompt_toolkit":
        raise
    _PROMPT_TOOLKIT_IMPORT_ERROR = exc
else:
    _PROMPT_TOOLKIT_IMPORT_ERROR = None

__all__ = [
    "ChatCompleter",
    "build_completions",
]

if _PROMPT_TOOLKIT_IMPORT_ERROR is None:
    __all__.extend(
        [
            "BodyScaffoldParts",
            "CHAT_STYLE",
            "ChatShell",
            "CommandPalette",
            "ConfirmationDialog",
            "ContextRail",
            "HelpOverlay",
            "OverlaySpec",
            "RepoContext",
            "ReferencePicker",
            "SessionRail",
            "anchored_overlay",
            "build_overlay_host",
            "positioned_overlay",
            "build_session_rail_window",
            "ShellScaffoldParts",
            "build_body_scaffold",
            "build_chat_key_bindings",
            "session_rail_fragments",
            "build_transcript_first_scaffold",
            "discover_repo_context",
        ]
    )
