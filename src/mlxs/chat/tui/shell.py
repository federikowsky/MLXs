"""Prompt-toolkit shell widget/layout host for interactive chat."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import ThreadedCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.containers import WindowAlign
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput

from mlxs.chat.input import parse_file_mentions
from mlxs.chat.present.chrome import (
    build_composer_context_fragments,
    build_composer_shortcuts_fragments,
    build_footer_left_fragments,
    build_footer_right_fragments,
    build_header_fragments,
    display_path,
    short_model_name,
    truncate_text,
)
from mlxs.chat.present.context import build_context_rail_summary
from mlxs.chat.present.commands import command_context, command_names, help_card
from mlxs.chat.present.progress import build_progress_fragments, progress_visible
from mlxs.chat.present.sessions import SessionListItem, item_from_session
from mlxs.chat.present.transcript import (
    TranscriptEntry,
    empty_state_fragments,
    entry_from_message,
    render_entry,
)
from mlxs.chat.session import ChatSession
from mlxs.chat.tui.completion import ChatCompleter, mention_completion_token
from mlxs.chat.tui.context import ContextRail
from mlxs.chat.tui.keymap import build_chat_key_bindings
from mlxs.chat.tui.overlay import anchored_overlay, positioned_overlay
from mlxs.chat.tui.palette import CommandPalette
from mlxs.chat.tui.reference_picker import ReferencePicker
from mlxs.chat.tui.rail import SessionRail
from mlxs.chat.tui.scaffold import (
    BodyScaffoldParts,
    ShellScaffoldParts,
    build_body_scaffold,
    build_transcript_first_scaffold,
)
from mlxs.chat.tui.style import CHAT_STYLE


@dataclass(frozen=True)
class RepoContext:
    """Stable repo information shown in the shell chrome."""

    cwd: Path
    cwd_label: str
    branch: str | None


def discover_repo_context(cwd: Path | None = None) -> RepoContext:
    """Collect cwd + git branch once at shell startup."""
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
        cwd_label=display_path(workdir),
        branch=branch,
    )


class ChatShell:
    """Docked prompt-toolkit shell for the interactive MLXs chat."""

    def __init__(
        self,
        model_id: str,
        session: ChatSession,
        *,
        max_tokens: int,
        temperature: float,
        repo: RepoContext | None = None,
    ) -> None:
        self._model_id = model_id or "default"
        self._session = session
        self._repo = repo or discover_repo_context()
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._message_entries: list[TranscriptEntry] = []
        self._notice_entries: list[TranscriptEntry] = []
        self._pending_assistant = ""
        self._submitted_inputs: list[str] = []
        self._history_index = 0
        self._lock = threading.Lock()
        self._state = "idle"
        self._state_detail = "Ready"
        self._submit_callback: Callable[[str], None] | None = None
        self._cancel_callback: Callable[[], None] | None = None
        self._previous_session_callback: Callable[[], None] | None = None
        self._next_session_callback: Callable[[], None] | None = None

        self._history = InMemoryHistory()
        self._buffer = Buffer(
            multiline=True,
            completer=ThreadedCompleter(ChatCompleter(command_names())),
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            history=self._history,
        )

        transcript_control = FormattedTextControl(
            text=self._transcript_fragments,
            focusable=False,
            get_cursor_position=self._transcript_cursor,
        )
        self._transcript_window = Window(
            content=transcript_control,
            wrap_lines=True,
            style="class:transcript",
            dont_extend_height=False,
        )
        self._session_rail = SessionRail()
        self._session_list_window = self._session_rail.container
        self._context_rail = ContextRail()
        self._context_window = self._context_rail.window
        self._command_palette = CommandPalette(invalidate=self._invalidate)
        self._reference_picker = ReferencePicker(cwd=lambda: self._repo.cwd, invalidate=self._invalidate)
        self._header_window = Window(
            content=FormattedTextControl(self._header_fragments),
            height=2,
            style="class:header",
        )
        self._meta_row = VSplit(
            [
                Window(
                    content=FormattedTextControl(self._composer_context_fragments),
                    height=1,
                    style="class:composer.context",
                ),
                Window(
                    content=FormattedTextControl(self._composer_shortcuts_fragments),
                    height=1,
                    style="class:composer.shortcuts",
                    align=WindowAlign.RIGHT,
                    width=Dimension(min=28),
                ),
            ],
            height=1,
        )
        self._footer = VSplit(
            [
                Window(
                    content=FormattedTextControl(self._footer_left_fragments),
                    height=1,
                    style="class:statusbar",
                ),
                Window(
                    content=FormattedTextControl(self._footer_right_fragments),
                    height=1,
                    style="class:statusbar",
                    align=WindowAlign.RIGHT,
                    width=Dimension(min=28),
                ),
            ],
            height=1,
        )
        self._input_window = Window(
            content=BufferControl(
                buffer=self._buffer,
                focus_on_click=True,
                input_processors=[BeforeInput([("class:composer.prompt", "> ")])],
            ),
            height=Dimension(min=3, max=8),
            style="class:composer",
        )
        self._progress_window = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(self._progress_fragments),
                height=1,
                style="class:progress",
            ),
            filter=Condition(lambda: progress_visible(state=self._state)),
        )

        container = build_transcript_first_scaffold(
            ShellScaffoldParts(
                header=self._header_window,
                body=build_body_scaffold(
                    BodyScaffoldParts(
                        left=self._session_list_window,
                        center=self._transcript_window,
                        right=self._context_window,
                    )
                ),
                progress=self._progress_window,
                composer=self._input_window,
                composer_meta=self._meta_row,
                footer=self._footer,
                overlays=(
                    anchored_overlay(
                        CompletionsMenu(max_height=10),
                        xcursor=True,
                        ycursor=True,
                    ),
                    positioned_overlay(
                        self._command_palette.container,
                        top=2,
                        left=8,
                        right=8,
                    ),
                    positioned_overlay(
                        self._reference_picker.container,
                        top=2,
                        left=8,
                        right=8,
                    ),
                ),
            )
        )

        self._application = Application(
            layout=Layout(container, focused_element=self._input_window),
            key_bindings=build_chat_key_bindings(
                buffer=self._buffer,
                get_state=lambda: self._state,
                get_cancel_callback=lambda: self._cancel_callback,
                is_filter_focused=lambda: self._application.layout.current_window
                is self._session_rail.filter_window,
                is_palette_open=lambda: self._command_palette.visible,
                is_palette_focused=lambda: self._application.layout.current_window
                is self._command_palette.filter_window,
                is_reference_picker_open=lambda: self._reference_picker.visible,
                is_reference_picker_focused=lambda: self._application.layout.current_window
                is self._reference_picker.filter_window,
                focus_filter=self._focus_filter,
                focus_composer=self._focus_composer,
                focus_palette=self._focus_palette,
                close_palette=self._close_palette,
                open_reference_picker=self._open_reference_picker,
                close_reference_picker=self._close_reference_picker,
                get_previous_session_callback=lambda: self._previous_session_callback,
                get_next_session_callback=lambda: self._next_session_callback,
                palette_previous=self._palette_previous,
                palette_next=self._palette_next,
                palette_accept=self._accept_palette,
                reference_previous=self._reference_previous,
                reference_next=self._reference_next,
                reference_accept=self._accept_reference,
                submit_buffer=self._submit_buffer,
                history_previous=self._history_previous,
                history_next=self._history_next,
                invalidate=self._invalidate,
            ),
            style=CHAT_STYLE,
            mouse_support=True,
            full_screen=True,
        )
        self.sync_session(session, clear_notices=True)

    def run(
        self,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], None],
        on_previous_session: Callable[[], None] | None = None,
        on_next_session: Callable[[], None] | None = None,
    ) -> None:
        self._submit_callback = on_submit
        self._cancel_callback = on_cancel
        self._previous_session_callback = on_previous_session
        self._next_session_callback = on_next_session
        self._application.run()

    def request_exit(self) -> None:
        self._application.exit()

    def _focus_filter(self) -> None:
        if self._command_palette.visible:
            self._command_palette.close()
        if self._reference_picker.visible:
            self._reference_picker.close()
        self._application.layout.focus(self._session_rail.filter_window)
        self._invalidate()

    def _focus_composer(self) -> None:
        if self._command_palette.visible:
            self._command_palette.close()
        if self._reference_picker.visible:
            self._reference_picker.close()
        self._application.layout.focus(self._input_window)
        self._invalidate()

    def _focus_palette(self) -> None:
        self._buffer.complete_state = None
        if self._reference_picker.visible:
            self._reference_picker.close()
        self._command_palette.open()
        self._application.layout.focus(self._command_palette.filter_window)
        self._invalidate()

    def _close_palette(self) -> None:
        self._command_palette.close()
        self._application.layout.focus(self._input_window)
        self._invalidate()

    def _open_reference_picker(self) -> None:
        mention_token = mention_completion_token(self._buffer.document.text_before_cursor)
        if mention_token is not None:
            self._buffer.complete_state = None
            if self._command_palette.visible:
                self._command_palette.close()
            cursor = self._buffer.cursor_position
            self._reference_picker.open(
                replace_start=cursor - len(mention_token),
                replace_end=cursor,
                initial_query=mention_token[1:],
            )
            self._application.layout.focus(self._reference_picker.filter_window)
            self._invalidate()
            return
        if not self._reference_trigger_allowed():
            text = self._buffer.text
            cursor = self._buffer.cursor_position
            new_text = f"{text[:cursor]}@{text[cursor:]}"
            self._buffer.document = Document(new_text, cursor_position=cursor + 1)
            return
        self._buffer.complete_state = None
        if self._command_palette.visible:
            self._command_palette.close()
        self._reference_picker.open(
            replace_start=self._buffer.cursor_position,
            replace_end=self._buffer.cursor_position,
        )
        self._application.layout.focus(self._reference_picker.filter_window)
        self._invalidate()

    def _close_reference_picker(self) -> None:
        self._reference_picker.close()
        self._application.layout.focus(self._input_window)
        self._invalidate()

    def _palette_previous(self) -> None:
        self._command_palette.move_selection(-1)

    def _palette_next(self) -> None:
        self._command_palette.move_selection(1)

    def _accept_palette(self) -> None:
        item = self._command_palette.selected_item()
        if item is None:
            self._close_palette()
            return
        text = item.insert_text
        self._buffer.document = Document(text, cursor_position=len(text))
        self._close_palette()

    def _reference_previous(self) -> None:
        self._reference_picker.move_selection(-1)

    def _reference_next(self) -> None:
        self._reference_picker.move_selection(1)

    def _accept_reference(self) -> None:
        item = self._reference_picker.selected_item()
        if item is None:
            self._close_reference_picker()
            return
        start = self._reference_picker.replace_start
        end = self._reference_picker.replace_end
        text = self._buffer.text
        insert_text = item.insert_text
        if end == len(text) or (end < len(text) and not text[end].isspace()):
            insert_text = f"{insert_text} "
        new_text = f"{text[:start]}{insert_text}{text[end:]}"
        cursor = start + len(insert_text)
        self._buffer.document = Document(new_text, cursor_position=cursor)
        self._close_reference_picker()

    def _reference_trigger_allowed(self) -> bool:
        if self._application.layout.current_window is not self._input_window:
            return False
        if self._buffer.text == "":
            return True
        before = self._buffer.document.text_before_cursor
        if not before:
            return True
        return before[-1].isspace()

    def sync_session(
        self,
        session: ChatSession,
        *,
        session_items: list[SessionListItem] | None = None,
        clear_notices: bool = False,
    ) -> None:
        with self._lock:
            self._session = session
            self._message_entries = [entry_from_message(message) for message in session.messages]
            self._session_rail.set_items(session_items or [item_from_session(session)])
            self._context_rail.set_summary(self._build_context_summary_locked())
            self._pending_assistant = ""
            if clear_notices:
                self._notice_entries.clear()
        self._invalidate()

    def set_state(self, state: str, detail: str) -> None:
        with self._lock:
            self._state = state
            self._state_detail = detail
            self._context_rail.set_summary(self._build_context_summary_locked())
        self._invalidate()

    def show_help(self) -> None:
        self._append_notice("help", "Commands", help_card())

    def show_status(self, text: str) -> None:
        self._append_notice("status", "Status", text)

    def show_error(self, text: str) -> None:
        self._append_notice("error", "Error", text)

    def show_stats(self, session: ChatSession) -> None:
        turns = sum(1 for msg in session.messages if msg.role == "user")
        assistant_messages = sum(1 for msg in session.messages if msg.role == "assistant")
        system_prompt = session.system_message()
        lines = [
            f"title: {session.title}",
            f"session: {session.session_id}",
            f"model: {self._model_id}",
            f"user turns: {turns}",
            f"assistant turns: {assistant_messages}",
            f"system prompt: {'on' if system_prompt else 'off'}",
        ]
        if system_prompt:
            lines.append(f"system preview: {truncate_text(system_prompt, 100)}")
        self._append_notice("status", "Session", "\n".join(lines))

    def show_runtime(self, *, max_tokens: int, temperature: float) -> None:
        lines = [
            f"model: {self._model_id}",
            f"max_tokens: {max_tokens}",
            f"temperature: {temperature}",
            f"cwd: {self._repo.cwd_label}",
            f"branch: {self._repo.branch or '-'}",
        ]
        self._append_notice("status", "Runtime", "\n".join(lines))

    def show_history(self, session: ChatSession, limit: int) -> None:
        transcript = [msg for msg in session.messages if msg.role != "system"]
        if not transcript:
            self._append_notice("status", "History", "No chat history yet.")
            return
        lines = []
        for message in transcript[-limit:]:
            role = {
                "assistant": "assistant",
                "user": "user",
            }.get(message.role, message.role)
            lines.append(f"{role}: {truncate_text(message.content, 120)}")
        self._append_notice("status", f"History ({min(limit, len(transcript))})", "\n".join(lines))

    def stream_reply(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._pending_assistant += text
        self._invalidate()

    def finish_reply(self, *, emitted_text: bool) -> None:
        if emitted_text:
            self._invalidate()

    def close_reply(self) -> None:
        with self._lock:
            self._pending_assistant = ""
        self._invalidate()

    def prompt(self) -> str:
        return ""

    def _append_notice(self, kind: str, title: str, body: str) -> None:
        with self._lock:
            self._notice_entries.append(TranscriptEntry(kind=kind, title=title, body=body))
            self._notice_entries = self._notice_entries[-24:]
        self._invalidate()

    def _submit_buffer(self) -> None:
        text = self._buffer.text
        self._buffer.document = Document("")
        self._buffer.complete_state = None
        self._history_index = len(self._submitted_inputs)
        if not text.strip():
            self._invalidate()
            return
        self._submitted_inputs.append(text)
        self._history.append_string(text)
        if self._submit_callback is not None:
            self._submit_callback(text)
        self._invalidate()

    def _history_previous(self) -> None:
        if not self._submitted_inputs or self._buffer.text:
            return
        if self._history_index == 0:
            return
        self._history_index -= 1
        text = self._submitted_inputs[self._history_index]
        self._buffer.document = Document(text, cursor_position=len(text))
        self._invalidate()

    def _history_next(self) -> None:
        if not self._submitted_inputs or self._buffer.text:
            return
        if self._history_index >= len(self._submitted_inputs) - 1:
            self._history_index = len(self._submitted_inputs)
            self._buffer.document = Document("")
            self._invalidate()
            return
        self._history_index += 1
        text = self._submitted_inputs[self._history_index]
        self._buffer.document = Document(text, cursor_position=len(text))
        self._invalidate()

    def _header_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            title = self._session.title
            turns = sum(1 for msg in self._session.messages if msg.role == "user")
            system_on = self._session.system_message() is not None
        return build_header_fragments(
            model_id=self._model_id,
            title=title,
            cwd_label=self._repo.cwd_label,
            branch=self._repo.branch,
            turns=turns,
            system_on=system_on,
        )

    def _composer_context_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            text = self._buffer.text
            state = self._state
            title = self._session.title
            system_on = self._session.system_message() is not None
        mentions = parse_file_mentions(text)
        return build_composer_context_fragments(
            text=text,
            state=state,
            title=title,
            system_on=system_on,
            palette_open=self._command_palette.visible,
            reference_picker_open=self._reference_picker.visible,
            command_context=command_context(text.strip()) if text.strip().startswith("/") else None,
            mention_paths=[match.path for match in mentions],
        )

    def _composer_shortcuts_fragments(self) -> list[tuple[str, str]]:
        return build_composer_shortcuts_fragments(
            state=self._state,
            palette_open=self._command_palette.visible,
            reference_picker_open=self._reference_picker.visible,
        )

    def _footer_left_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            turns = sum(1 for msg in self._session.messages if msg.role == "user")
            system_on = self._session.system_message() is not None
        return build_footer_left_fragments(
            model_id=self._model_id,
            branch=self._repo.branch,
            turns=turns,
            system_on=system_on,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def _footer_right_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            state = self._state
            detail = self._state_detail
        return build_footer_right_fragments(state=state, detail=detail)

    def _progress_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            state = self._state
            detail = self._state_detail
        return build_progress_fragments(state=state, detail=detail)

    def _build_context_summary_locked(self):
        return build_context_rail_summary(
            session=self._session,
            model_id=self._model_id,
            state=self._state,
            detail=self._state_detail,
        )

    def _transcript_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            entries = [*self._message_entries, *self._notice_entries]
            pending = self._pending_assistant

        fragments: list[tuple[str, str]] = []
        for index, entry in enumerate(entries):
            if index:
                fragments.append(("", "\n"))
            fragments.extend(render_entry(entry))

        if pending:
            if entries:
                fragments.append(("", "\n"))
            fragments.extend(
                render_entry(
                    TranscriptEntry(kind="assistant", title="assistant", body=pending),
                    pending=True,
                )
            )

        if not fragments:
            fragments.extend(empty_state_fragments())
        return fragments

    def _session_list_fragments(self) -> list[tuple[str, str]]:
        return self._session_rail.fragments()

    def _transcript_cursor(self) -> Point:
        text = "".join(part for _, part in self._transcript_fragments())
        lines = text.splitlines() or [""]
        return Point(x=len(lines[-1]), y=len(lines) - 1)

    def _invalidate(self) -> None:
        try:
            self._application.invalidate()
        except Exception:
            return
__all__ = ["ChatShell", "RepoContext", "discover_repo_context"]
