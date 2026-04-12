from __future__ import annotations

from pathlib import Path

from mlxs.chat.tui.shell import ChatShell, RepoContext
from mlxs.chat.session import ChatSession


def test_chat_shell_uses_full_screen_application() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    assert shell._application.full_screen is True


def test_chat_shell_uses_multiline_composer_buffer() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    assert shell._buffer.multiline() is True
