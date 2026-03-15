"""Transcript rendering for the chat shell.

Extracted from cli.py to keep file size under ~800 lines (AC7).
"""

from __future__ import annotations

from mlxs.chat.session import ChatMessage


class _TranscriptEntry:
    """One rendered entry in the chat transcript."""

    __slots__ = ("attachments", "body", "kind", "title")

    def __init__(
        self,
        kind: str,
        title: str,
        body: str,
        attachments: tuple[str, ...] = (),
    ) -> None:
        self.kind = kind
        self.title = title
        self.body = body
        self.attachments = attachments


def entry_from_message(message: ChatMessage) -> _TranscriptEntry:
    """Convert a ChatMessage to a transcript entry."""
    attachments = tuple(
        attachment["path"]
        for attachment in message.metadata.get("attachments", [])
        if isinstance(attachment, dict) and isinstance(attachment.get("path"), str)
    )
    kind = message.role if message.role in {"user", "assistant", "system"} else "status"
    return _TranscriptEntry(
        kind=kind,
        title=kind,
        body=message.content,
        attachments=attachments,
    )


def render_entry(entry: _TranscriptEntry, *, pending: bool = False) -> list[tuple[str, str]]:
    """Render a transcript entry to prompt-toolkit formatted text fragments."""
    if entry.kind == "user":
        return _render_user_entry(entry)
    if entry.kind == "assistant":
        return _render_assistant_entry(entry, pending=pending)

    label_style = {
        "error": "class:label.error",
        "help": "class:label.help",
        "status": "class:label.status",
        "system": "class:label.system",
    }.get(entry.kind, "class:label.status")
    body_style = {
        "error": "class:body.error",
        "help": "class:body.help",
        "status": "class:body.status",
        "system": "class:body.system",
    }.get(entry.kind, "class:body.status")
    title = {
        "error": "error",
        "help": "help",
        "status": "note",
        "system": "system",
    }.get(entry.kind, entry.title)

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


def _render_user_entry(entry: _TranscriptEntry) -> list[tuple[str, str]]:
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
    entry: _TranscriptEntry,
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


def empty_state_fragments() -> list[tuple[str, str]]:
    """Fragments shown when the transcript is empty."""
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
