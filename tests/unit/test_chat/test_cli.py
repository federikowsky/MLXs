"""Tests for the prompt-toolkit chat shell helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.document import Document

from mlxs.chat.cli import (
    AttachmentResolutionError,
    ChatShell,
    RepoContext,
    build_completions,
    parse_file_mentions,
    resolve_attachments,
)
from mlxs.chat.session import ChatSession


def test_build_completions_matches_slash_commands() -> None:
    commands = ("/help", "/history", "/export")

    assert build_completions("/h", "/h", commands) == ["/help", "/history"]


def test_build_completions_matches_export_paths(tmp_path: Path) -> None:
    commands = ("/help", "/history", "/export")
    (tmp_path / "chat.md").write_text("", encoding="utf-8")
    (tmp_path / "chat-notes.md").write_text("", encoding="utf-8")

    matches = build_completions("/export chat", "chat", commands, cwd=tmp_path)

    assert matches == ["chat-notes.md", "chat.md"]


def test_build_completions_matches_file_mentions(tmp_path: Path) -> None:
    commands = ("/help", "/history", "/export")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    matches = build_completions("@src/m", "@src/m", commands, cwd=tmp_path)

    assert matches == ["@src/main.py"]


def test_parse_file_mentions_supports_plain_and_quoted_paths() -> None:
    matches = parse_file_mentions('check @src/app.py and @"docs/cli arguments.md"')

    assert [match.path for match in matches] == ["src/app.py", "docs/cli arguments.md"]


def test_resolve_attachments_reads_text_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("print('hi')\n", encoding="utf-8")

    mention = source.relative_to(tmp_path)
    attachments = resolve_attachments(f"review @{mention} @{mention}", cwd=tmp_path)

    assert len(attachments) == 1
    assert attachments[0].path == "src/app.py"
    assert attachments[0].content == "print('hi')\n"


def test_resolve_attachments_rejects_binary_files(tmp_path: Path) -> None:
    binary = tmp_path / "image.bin"
    binary.write_bytes(b"\x00\x01\x02")

    with pytest.raises(AttachmentResolutionError):
        resolve_attachments("@image.bin", cwd=tmp_path)


def test_chat_shell_renders_help_and_pending_stream() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell.show_help()
    shell.set_state("generating", "Streaming reply")
    shell.stream_reply("Hello")
    shell.stream_reply(" world")

    transcript = "".join(part for _, part in shell._transcript_fragments())
    footer = "".join(part for _, part in shell._footer_right_fragments())

    assert "slash commands:" in transcript
    assert "live" in transcript
    assert "Hello world" in transcript
    assert "Streaming reply" in footer
    assert "Streaming" in footer


def test_chat_shell_renders_empty_state_and_dynamic_context() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    transcript = "".join(part for _, part in shell._transcript_fragments())
    context = "".join(part for _, part in shell._composer_context_fragments())
    header = "".join(part for _, part in shell._header_fragments())

    assert "What can I help you ship today?" in transcript
    assert "review @src/mlxs/server/chat.py" in transcript
    assert "working in New chat" in context
    assert "system off" in context
    assert "Qwen3.5-2B-PARO" in header
    assert "main" in header

    shell._buffer.document = Document("/re", cursor_position=3)
    command_context = "".join(part for _, part in shell._composer_context_fragments())
    assert "/retry" in command_context

    shell._buffer.document = Document("@src/server.py", cursor_position=len("@src/server.py"))
    attachment_context = "".join(part for _, part in shell._composer_context_fragments())
    footer = "".join(part for _, part in shell._footer_left_fragments())
    shortcuts = "".join(part for _, part in shell._composer_shortcuts_fragments())

    assert "1 file reference(s) ready" in attachment_context
    assert "0 turns" in footer
    assert "/ palette" in shortcuts
    assert "@ picker" in shortcuts
