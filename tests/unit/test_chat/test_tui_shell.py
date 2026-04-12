from __future__ import annotations

from pathlib import Path

from prompt_toolkit.document import Document

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


def test_chat_shell_show_help_opens_overlay_without_transcript_notice() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell.show_help()

    transcript = "".join(part for _, part in shell._transcript_fragments())
    help_overlay = "".join(part for _, part in shell._help_overlay.fragments())

    assert "slash commands:" not in transcript
    assert "slash commands:" in help_overlay
    assert shell._help_overlay.visible is True


def test_chat_shell_confirmation_accept_runs_callback() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )
    called: list[str] = []

    shell.show_confirmation(
        title="Clear conversation?",
        body="This removes the current transcript.",
        confirm_label="Clear",
        on_confirm=lambda: called.append("confirmed"),
    )
    shell._accept_confirmation()

    assert called == ["confirmed"]
    assert shell._confirmation_dialog.visible is False


def test_chat_shell_confirmation_cancel_closes_dialog() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell.show_confirmation(
        title="Clear conversation?",
        body="This removes the current transcript.",
        confirm_label="Clear",
        on_confirm=lambda: None,
    )
    shell._cancel_confirmation()

    assert shell._confirmation_dialog.visible is False


def test_chat_shell_context_and_shortcuts_reflect_confirmation_mode() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell.show_confirmation(
        title="Clear conversation?",
        body="This removes the current transcript.",
        confirm_label="Clear",
        on_confirm=lambda: None,
    )

    context = "".join(part for _, part in shell._composer_context_fragments())
    shortcuts = "".join(part for _, part in shell._composer_shortcuts_fragments())

    assert "confirmation open" in context
    assert "Enter confirm" in shortcuts


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


def test_chat_shell_renders_compact_context_rail_summary() -> None:
    session = ChatSession(
        title="Latency hypothesis",
        model_path="z-lab/Qwen3.5-2B-PARO",
    )
    session.add_user_message("Check current context")
    session.updated_at = "2026-04-12T12:34:00+00:00"
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell.set_state("generating", "Streaming reply")
    rendered = "".join(part for _, part in shell._context_rail.fragments())

    assert "Context" in rendered
    assert "Latency hypothesis" in rendered
    assert "updated" in rendered
    assert "turns" in rendered
    assert "Running" in rendered
    assert "Streaming reply" in rendered


def test_chat_shell_run_does_not_append_startup_status_notice() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell._application.run = lambda *args, **kwargs: None
    shell.run(on_submit=lambda text: None, on_cancel=lambda: None)

    assert shell._notice_entries == []


def test_chat_shell_opens_palette_and_inserts_selected_command() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell._focus_palette()
    shell._command_palette.filter_buffer.text = "hist"
    shell._accept_palette()

    assert shell._command_palette.visible is False
    assert shell._buffer.text == "/history "
    assert shell._application.layout.current_window is shell._input_window


def test_chat_shell_context_and_shortcuts_reflect_palette_mode() -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=Path("/tmp/project"), cwd_label="~/project", branch="main"),
    )

    shell._focus_palette()

    context = "".join(part for _, part in shell._composer_context_fragments())
    shortcuts = "".join(part for _, part in shell._composer_shortcuts_fragments())

    assert "command palette open" in context
    assert "Type to filter" in shortcuts


def test_chat_shell_opens_reference_picker_and_inserts_selected_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=tmp_path, cwd_label="~/project", branch="main"),
    )

    shell._open_reference_picker()
    shell._reference_picker.filter_buffer.text = "src/m"
    shell._accept_reference()

    assert shell._reference_picker.visible is False
    assert shell._buffer.text == "@src/main.py "
    assert shell._application.layout.current_window is shell._input_window


def test_chat_shell_context_and_shortcuts_reflect_reference_picker_mode(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=tmp_path, cwd_label="~/project", branch="main"),
    )

    shell._open_reference_picker()

    context = "".join(part for _, part in shell._composer_context_fragments())
    shortcuts = "".join(part for _, part in shell._composer_shortcuts_fragments())

    assert "reference picker open" in context
    assert "Type to filter" in shortcuts


def test_chat_shell_reference_trigger_falls_back_to_literal_at_inside_word(tmp_path: Path) -> None:
    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=tmp_path, cwd_label="~/project", branch="main"),
    )

    shell._buffer.document = Document("email", cursor_position=5)
    shell._open_reference_picker()

    assert shell._reference_picker.visible is False
    assert shell._buffer.text == "email@"


def test_chat_shell_reference_insert_does_not_duplicate_trailing_space(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=tmp_path, cwd_label="~/project", branch="main"),
    )

    shell._buffer.document = Document("review ", cursor_position=len("review "))
    shell._open_reference_picker()
    shell._reference_picker.filter_buffer.text = "src/m"
    shell._accept_reference()

    assert shell._buffer.text == "review @src/main.py "


def test_chat_shell_reference_insert_adds_space_before_attached_text(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=tmp_path, cwd_label="~/project", branch="main"),
    )

    shell._buffer.document = Document("review later", cursor_position=len("review "))
    shell._open_reference_picker()
    shell._reference_picker.filter_buffer.text = "src/m"
    shell._accept_reference()

    assert shell._buffer.text == "review @src/main.py later"


def test_chat_shell_reference_picker_reuses_existing_partial_mention(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=tmp_path, cwd_label="~/project", branch="main"),
    )

    shell._buffer.document = Document("review @src/m", cursor_position=len("review @src/m"))
    shell._open_reference_picker()

    assert shell._reference_picker.visible is True
    assert shell._reference_picker.filter_buffer.text == "src/m"

    shell._accept_reference()

    assert shell._buffer.text == "review @src/main.py "


def test_chat_shell_reference_picker_reuses_existing_quoted_partial_mention(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "read me.md").write_text("hello\n", encoding="utf-8")

    session = ChatSession(model_path="z-lab/Qwen3.5-2B-PARO")
    shell = ChatShell(
        "z-lab/Qwen3.5-2B-PARO",
        session,
        max_tokens=256,
        temperature=0.7,
        repo=RepoContext(cwd=tmp_path, cwd_label="~/project", branch="main"),
    )

    shell._buffer.document = Document('review @"docs/read', cursor_position=len('review @"docs/read'))
    shell._open_reference_picker()

    assert shell._reference_picker.visible is True
    assert shell._reference_picker.filter_buffer.text == '"docs/read'


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


def test_chat_shell_focuses_filter_and_filters_session_rail() -> None:
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
            session_id="conv-a",
            title="Latency hypothesis",
            model_path=session.model_path,
            updated_at="2026-04-12T12:00:00+00:00",
            message_count=1,
            active=True,
        ),
        ChatSessionSummary(
            session_id="conv-b",
            title="Quantum basics",
            model_path=session.model_path,
            updated_at="2026-04-12T12:01:00+00:00",
            message_count=2,
            active=False,
        ),
    ]
    shell.sync_session(session, session_items=[item_from_summary(summary) for summary in summaries])

    shell._focus_filter()
    shell._session_rail.filter_buffer.text = "quant"
    rendered = "".join(part for _, part in shell._session_list_fragments())

    assert shell._application.layout.current_window is shell._session_rail.filter_window
    assert "Quantum basics" in rendered
    assert "Latency hypothesis" not in rendered
