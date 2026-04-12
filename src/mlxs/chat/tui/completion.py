"""Prompt-toolkit completer and pure completion helpers for chat UI.

This module owns prompt-toolkit specific completion behavior while keeping the
logic testable via the pure ``build_completions`` helper.
"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document


class ChatCompleter(Completer):
    """Completion menu for slash commands, /export paths, and @file mentions."""

    def __init__(self, command_names: tuple[str, ...]) -> None:
        self._command_names = command_names

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ):
        del complete_event
        before = document.text_before_cursor
        stripped = before.lstrip()

        if stripped.startswith("/") and " " not in stripped[1:]:
            for candidate in self._command_names:
                if candidate.startswith(stripped):
                    yield Completion(
                        candidate,
                        start_position=-len(stripped),
                        display=candidate,
                    )
            return

        export_prefix = _export_completion_prefix(before)
        if export_prefix is not None:
            for candidate in _path_completion_values(export_prefix, Path.cwd()):
                yield Completion(
                    candidate,
                    start_position=-len(export_prefix),
                    display=candidate,
                    display_meta="export path",
                )
            return

        mention_token = mention_completion_token(before)
        if mention_token is None:
            return
        mention_prefix = mention_token[1:]
        for candidate in reference_candidate_values(mention_prefix, Path.cwd()):
            yield Completion(
                f"@{candidate}",
                start_position=-len(mention_token),
                display=f"@{candidate}",
                display_meta="attach file",
            )


def build_completions(
    buffer: str,
    text: str,
    command_names: tuple[str, ...],
    *,
    cwd: Path | None = None,
) -> list[str]:
    """Pure completion helper used by unit tests and non-widget callers."""
    workdir = cwd or Path.cwd()
    stripped = buffer.lstrip()
    if stripped.startswith("/") and " " not in stripped[1:]:
        return [name for name in command_names if name.startswith(text)]

    export_prefix = _export_completion_prefix(buffer)
    if export_prefix is not None:
        return _path_completion_values(export_prefix, workdir)

    mention_token = mention_completion_token(buffer)
    if mention_token is not None:
        return [f"@{value}" for value in reference_candidate_values(mention_token[1:], workdir)]

    return []


def _path_completion_values(raw_prefix: str, cwd: Path) -> list[str]:
    prefix = raw_prefix or ""
    quoted = prefix.startswith('"')
    prefix = prefix[1:] if quoted else prefix

    expanded = Path(prefix).expanduser()
    if expanded.is_absolute():
        base_dir = expanded.parent
        stem = expanded.name
    else:
        base_dir = (cwd / expanded).parent
        stem = expanded.name
    if not base_dir.exists():
        return []

    matches: list[str] = []
    for path in sorted(base_dir.iterdir()):
        if not path.name.startswith(stem):
            continue
        try:
            display = str(path.relative_to(cwd))
        except ValueError:
            display = str(path)
        if path.is_dir():
            display += "/"
        if " " in display or quoted:
            display = f'"{display}"'
        matches.append(display)
    return matches


def reference_candidate_values(raw_prefix: str, cwd: Path) -> list[str]:
    """Return file-only candidate values for @ reference insertion."""
    values = _path_completion_values(raw_prefix, cwd)
    return [value for value in values if not value.endswith("/")]


def _export_completion_prefix(buffer: str) -> str | None:
    stripped = buffer.lstrip()
    if not stripped.startswith("/export "):
        return None
    return stripped[len("/export ") :]


def mention_completion_token(buffer: str) -> str | None:
    before = buffer.rstrip("\n")
    for start in range(len(before) - 1, -1, -1):
        if before[start] != "@":
            continue
        if start > 0 and not before[start - 1].isspace():
            continue
        token = before[start:]
        if token == "@":
            return None
        if any(char.isspace() for char in token[1:]) and not token.startswith('@"'):
            return None
        return token
    return None


__all__ = [
    "ChatCompleter",
    "build_completions",
    "mention_completion_token",
    "reference_candidate_values",
]
