"""Transcript entry model and prompt-toolkit fragment rendering.

Single canonical location for:
  - ``TranscriptEntry`` — the immutable view model for one transcript row.
  - ``entry_from_message`` — maps a ``ChatMessage`` to a ``TranscriptEntry``.
  - ``render_entry`` — renders a ``TranscriptEntry`` to prompt-toolkit fragments.
  - ``empty_state_fragments`` — fragments shown when the transcript is empty.

No prompt_toolkit container or widget creation here — only fragment lists
(``list[tuple[str, str]]``).  All style class names are stable strings so
callers can override them via the style layer.

Authority: presentation layer — no I/O, no business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlxs.chat.session import ChatMessage


@dataclass(slots=True)
class TranscriptEntry:
    """One rendered entry in the chat transcript."""

    kind: str                          # "user" | "assistant" | "system" | "status" | "error" | "help"
    title: str                         # display title (e.g. "user", "error")
    body: str                          # message body text
    attachments: tuple[str, ...] = field(default_factory=tuple)


def entry_from_message(message: ChatMessage) -> TranscriptEntry:
    """Convert a ``ChatMessage`` to a ``TranscriptEntry``."""
    attachments = tuple(
        attachment["path"]
        for attachment in message.metadata.get("attachments", [])
        if isinstance(attachment, dict) and isinstance(attachment.get("path"), str)
    )
    kind = message.role if message.role in {"user", "assistant", "system"} else "status"
    return TranscriptEntry(
        kind=kind,
        title=kind,
        body=message.content,
        attachments=attachments,
    )


def render_entry(
    entry: TranscriptEntry,
    *,
    pending: bool = False,
) -> list[tuple[str, str]]:
    """Render *entry* to a list of prompt-toolkit ``(style, text)`` fragments.

    Parameters
    ----------
    entry:
        The entry to render.
    pending:
        When ``True`` the entry is the in-progress assistant streaming reply
        and a ``" live "`` badge is prepended.
    """
    if entry.kind == "user":
        return _render_user_entry(entry)
    if entry.kind == "assistant":
        return _render_assistant_entry(entry, pending=pending)
    return _render_notice_entry(entry)


def empty_state_fragments() -> list[tuple[str, str]]:
    """Return fragments shown when the transcript contains no messages."""
    return [
        ("class:label.status", " ready "),
        ("class:body.status", "  What can I help you ship today?\n"),
        (
            "class:body.status",
            "  Ask for code, inspect files with @path, or use /help to explore commands.\n",
        ),
        ("class:attachment", "  Examples\n"),
        ("class:body.status", "    /help\n"),
        ("class:body.status", "    /system You are a concise code reviewer\n"),
        ("class:body.status", "    review @src/mlxs/server/chat.py\n"),
        ("class:body.status", "    explain the current architecture of this repo"),
    ]


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------

_LABEL_STYLE: dict[str, str] = {
    "error": "class:label.error",
    "help": "class:label.help",
    "status": "class:label.status",
    "system": "class:label.system",
}
_BODY_STYLE: dict[str, str] = {
    "error": "class:body.error",
    "help": "class:body.help",
    "status": "class:body.status",
    "system": "class:body.system",
}
_TITLE_LABEL: dict[str, str] = {
    "error": "error",
    "help": "help",
    "status": "note",
    "system": "system",
}


def _render_notice_entry(entry: TranscriptEntry) -> list[tuple[str, str]]:
    label_style = _LABEL_STYLE.get(entry.kind, "class:label.status")
    body_style = _BODY_STYLE.get(entry.kind, "class:body.status")
    title = _TITLE_LABEL.get(entry.kind, entry.title)

    lines = entry.body.splitlines() or [""]
    fragments: list[tuple[str, str]] = [(label_style, f" {title} ")]
    if len(lines) == 1 and not entry.attachments:
        fragments.append((body_style, f"  {lines[0]}"))
        return fragments

    fragments.append(("", "\n"))
    for line in lines:
        fragments.append((body_style, f"  {line}\n"))
    if entry.attachments:
        for path in entry.attachments:
            fragments.append(("class:attachment", f"  @ {path}\n"))
    if fragments[-1][1].endswith("\n"):
        fragments[-1] = (fragments[-1][0], fragments[-1][1].rstrip("\n"))
    return fragments


def _render_user_entry(entry: TranscriptEntry) -> list[tuple[str, str]]:
    lines = entry.body.splitlines() or [""]
    fragments: list[tuple[str, str]] = [("class:label.user", f"> {lines[0]}")]
    for line in lines[1:]:
        fragments.append(("", "\n"))
        fragments.append(("class:body.user", f"  {line}"))
    for path in entry.attachments:
        fragments.append(("", "\n"))
        fragments.append(("class:attachment", f"  @ {path}"))
    return fragments


def _render_assistant_entry(
    entry: TranscriptEntry,
    *,
    pending: bool,
) -> list[tuple[str, str]]:
    lines = entry.body.splitlines() or [""]
    fragments: list[tuple[str, str]] = []
    if pending:
        fragments.append(("class:label.assistant", " live "))
        fragments.append(("", "\n"))
    for index, line in enumerate(lines):
        if index:
            fragments.append(("", "\n"))
        fragments.append(("class:body.assistant", line))
    return fragments
