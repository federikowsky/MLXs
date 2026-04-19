"""Compatibility facade for chat shell and helper modules.

This module owns:
  - ``parse_command`` — parse a ``/command`` line for the shell controller.

Pure helpers that have been extracted to dedicated modules are re-exported
here for backwards compatibility with existing callers and tests:

  Attachment parsing   → ``mlxs.chat.input``
  Command metadata     → ``mlxs.chat.present.commands``
  Transcript rendering → ``mlxs.chat.present.transcript``
  Completion logic     → ``mlxs.chat.tui.completion``
  Shell/layout         → ``mlxs.chat.tui.shell``
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# ── Canonical modules — re-exported for backwards compatibility ───────────────
from mlxs.chat.input import (  # noqa: F401
    AttachmentResolutionError,
    MentionMatch,
    ResolvedAttachment,
    parse_file_mentions,
    resolve_attachments,
)
from mlxs.chat.present.commands import (  # noqa: F401
    COMMANDS,
    CommandSpec,
    command_names,
    help_card,
)
from mlxs.chat.present.transcript import (
    TranscriptEntry as _TranscriptEntry,
    empty_state_fragments as _empty_state_fragments,
    entry_from_message as _entry_from_message,
    render_entry as _render_entry,
)
from mlxs.chat.session import ChatMessage, ChatSession
from mlxs.chat.tui.completion import (  # noqa: F401
    ChatCompleter as _ChatCompleter,
    build_completions,
)
try:
    from mlxs.chat.tui.shell import ChatShell, RepoContext, discover_repo_context
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in product-surface tests
    if exc.name != "prompt_toolkit":
        raise
    _PROMPT_TOOLKIT_IMPORT_ERROR = exc

    @dataclass(frozen=True)
    class RepoContext:
        """Fallback repo information when prompt-toolkit is unavailable."""

        cwd: Path
        cwd_label: str
        branch: str | None

    def discover_repo_context(cwd: Path | None = None) -> RepoContext:
        """Collect cwd + git branch even when the TUI dependency is unavailable."""
        workdir = cwd or Path.cwd()
        branch: str | None = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            candidate = result.stdout.strip()
            if result.returncode == 0 and candidate and candidate != "HEAD":
                branch = candidate
        except Exception:
            branch = None
        return RepoContext(
            cwd=workdir,
            cwd_label=str(workdir),
            branch=branch,
        )

    class ChatShell:
        """Fallback shell stub that raises a clear optional-dependency error."""

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise ModuleNotFoundError(
                "prompt_toolkit is required for ChatShell"
            ) from _PROMPT_TOOLKIT_IMPORT_ERROR

# ── Module-level helpers ──────────────────────────────────────────────────────


def parse_command(line: str) -> tuple[str, str] | None:
    """Parse an interactive slash command.

    Returns ``(name, args)`` or ``None`` if the line does not start with ``/``.
    """
    if not line.startswith("/"):
        return None
    body = line[1:].strip()
    if not body:
        return "", ""
    name, _, rest = body.partition(" ")
    return name.lower(), rest.strip()

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
