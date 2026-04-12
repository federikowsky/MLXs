from __future__ import annotations

from pathlib import Path

from mlxs.chat.present.sessions import item_from_summary
from mlxs.chat.tui.shell import ChatShell, RepoContext
from mlxs.chat.session import ChatSession
from mlxs.chat.store import ChatSessionSummary


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


def test_chat_shell_progress_fragments_reflect_generating_state() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell.set_state("generating", "Streaming reply")
    rendered = "".join(part for _, part in shell._progress_fragments())

    assert "Running" in rendered
    assert "Streaming reply" in rendered


def test_chat_shell_renders_passive_session_rail_items() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    summaries = [
        ChatSessionSummary(
            session_id=session.session_id,
            title="Latency hypothesis",
            model_path=session.model_path,
            updated_at="2026-04-12T12:00:00+00:00",
            message_count=1,
            active=True,
        )
    ]
    shell.sync_session(session, session_items=[item_from_summary(summary) for summary in summaries])

    rendered = "".join(part for _, part in shell._session_list_fragments())

    assert "Conversations" in rendered
    assert "Latency hypothesis" in rendered
