from __future__ import annotations

from pathlib import Path

from mlxs.chat.present.commands import command_names
from mlxs.chat.tui.completion import build_completions


def test_build_completions_matches_slash_commands() -> None:
    commands = command_names()
    assert build_completions("/h", "/h", commands) == ["/help", "/history"]


def test_build_completions_matches_export_paths(tmp_path: Path) -> None:
    commands = command_names()
    (tmp_path / "chat.md").write_text("", encoding="utf-8")
    (tmp_path / "chat-notes.md").write_text("", encoding="utf-8")

    matches = build_completions("/export chat", "chat", commands, cwd=tmp_path)

    assert matches == ["chat-notes.md", "chat.md"]


def test_build_completions_matches_file_mentions(tmp_path: Path) -> None:
    commands = command_names()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    matches = build_completions("@src/m", "@src/m", commands, cwd=tmp_path)

    assert matches == ["@src/main.py"]
