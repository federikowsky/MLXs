"""Attachment parsing and file mention resolution for the chat shell.

Extracted from cli.py to keep file size under ~800 lines (AC7).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedAttachment:
    """Resolved attachment payload from an `@file` mention."""

    path: str
    absolute_path: str
    content: str

    def to_metadata(self) -> dict[str, str]:
        return {
            "path": self.path,
            "absolute_path": self.absolute_path,
            "content": self.content,
        }


@dataclass(frozen=True)
class MentionMatch:
    """One parsed `@file` mention in the composer text."""

    raw: str
    path: str
    start: int
    end: int


class AttachmentResolutionError(ValueError):
    """Raised when an `@file` mention cannot be attached."""


def parse_file_mentions(text: str) -> list[MentionMatch]:
    """Parse `@file` and `@\"quoted path\"` mentions from a composer line."""
    matches: list[MentionMatch] = []
    index = 0
    while index < len(text):
        at = text.find("@", index)
        if at < 0:
            break
        if at > 0 and not text[at - 1].isspace():
            index = at + 1
            continue
        if at + 1 >= len(text):
            break
        if text[at + 1] == '"':
            end = at + 2
            parts: list[str] = []
            while end < len(text) and text[end] != '"':
                parts.append(text[end])
                end += 1
            if end >= len(text):
                index = at + 1
                continue
            path = "".join(parts).strip()
            if path:
                matches.append(
                    MentionMatch(raw=text[at : end + 1], path=path, start=at, end=end + 1)
                )
            index = end + 1
            continue
        end = at + 1
        while end < len(text) and not text[end].isspace():
            end += 1
        path = text[at + 1 : end].rstrip(",.;:")
        if path:
            matches.append(MentionMatch(raw=text[at:end], path=path, start=at, end=end))
        index = end
    return matches


def resolve_attachments(text: str, *, cwd: Path | None = None) -> list[ResolvedAttachment]:
    """Resolve and read all valid `@file` mentions in a composer line."""
    workdir = cwd or Path.cwd()
    seen: set[Path] = set()
    attachments: list[ResolvedAttachment] = []
    for mention in parse_file_mentions(text):
        raw_path = Path(mention.path).expanduser()
        resolved = raw_path if raw_path.is_absolute() else workdir / raw_path
        resolved = resolved.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise AttachmentResolutionError(f"Attachment not found: {mention.path}")
        if not resolved.is_file():
            raise AttachmentResolutionError(f"Attachment is not a file: {mention.path}")

        data = resolved.read_bytes()
        if b"\x00" in data:
            raise AttachmentResolutionError(
                f"Binary attachments are not supported: {mention.path}"
            )
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AttachmentResolutionError(
                f"Attachment is not valid UTF-8 text: {mention.path}"
            ) from exc

        try:
            display_path = str(resolved.relative_to(workdir))
        except ValueError:
            display_path = str(resolved)
        attachments.append(
            ResolvedAttachment(
                path=display_path,
                absolute_path=str(resolved),
                content=content,
            )
        )
        seen.add(resolved)
    return attachments
