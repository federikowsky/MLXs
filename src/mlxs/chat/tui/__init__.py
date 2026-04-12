"""Prompt-toolkit UI helpers for chat product surfaces."""

from mlxs.chat.tui.completion import ChatCompleter, build_completions
from mlxs.chat.tui.keymap import build_chat_key_bindings
from mlxs.chat.tui.shell import ChatShell, RepoContext, discover_repo_context
from mlxs.chat.tui.style import CHAT_STYLE

__all__ = [
    "CHAT_STYLE",
    "ChatCompleter",
    "ChatShell",
    "RepoContext",
    "build_chat_key_bindings",
    "build_completions",
    "discover_repo_context",
]
