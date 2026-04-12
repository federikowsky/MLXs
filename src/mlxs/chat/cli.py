"""Prompt-toolkit shell host and public compatibility re-exports for MLXs chat.

This module owns:
  - ``ChatShell`` — the docked prompt-toolkit shell widget/layout.
  - ``RepoContext`` — stable repo info shown in the shell chrome.
  - ``discover_repo_context`` — collect cwd + git branch once at startup.
  - ``parse_command`` — parse a ``/command`` line for the shell controller.

Pure helpers that have been extracted to dedicated modules are re-exported
here for backwards compatibility with existing callers and tests:

  Attachment parsing   → ``mlxs.chat.input``
  Command metadata     → ``mlxs.chat.present.commands``
  Transcript rendering → ``mlxs.chat.present.transcript``
  Completion logic     → ``mlxs.chat.tui.completion``
"""

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
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import WindowAlign
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.styles import Style

# ── Canonical modules — re-exported for backwards compatibility ───────────────
from mlxs.chat.input import (  # noqa: F401
    AttachmentResolutionError,
    MentionMatch,
    ResolvedAttachment,
    parse_file_mentions,
    resolve_attachments,
)
from mlxs.chat.present.commands import (  # noqa: F401
    COMMANDS,
    CommandSpec,
    command_names,
    command_context as _command_context,
    help_card,
)
from mlxs.chat.present.transcript import (
    TranscriptEntry as _TranscriptEntry,
    empty_state_fragments as _empty_state_fragments,
    entry_from_message as _entry_from_message,
    render_entry as _render_entry,
)
from mlxs.chat.session import ChatMessage, ChatSession
from mlxs.chat.tui.completion import (  # noqa: F401
    ChatCompleter as _ChatCompleter,
    build_completions,
)

# ── Style ─────────────────────────────────────────────────────────────────────

_STYLE = Style.from_dict(
    {
        "shell": "bg:#0b0f14 #e6edf3",
        "surface": "bg:#0e141b",
        "header": "bg:#0e141b #e6edf3",
        "header.brand": "bg:#ff8a65 #0b0f14 bold",
        "header.model": "bg:#0e141b #f0f6fc bold",
        "header.meta": "bg:#0e141b #8b949e",
        "header.path": "bg:#0e141b #c9d1d9",
        "header.badge": "bg:#18212b #cdd9e5 bold",
        "transcript": "bg:#0b0f14 #e6edf3",
        "label.user": "#cbd5e1 bold",
        "label.assistant": "bg:#163225 #b8f7c0 bold",
        "label.system": "bg:#4f4017 #f5d06b bold",
        "label.status": "bg:#162a40 #9fd3ff bold",
        "label.error": "bg:#4a1f1f #ffb4ae bold",
        "label.help": "bg:#36224d #e2c8ff bold",
        "body.user": "#dbe4ee",
        "body.assistant": "#edf5ff",
        "body.system": "#f4d57a",
        "body.status": "#c9d1d9",
        "body.error": "#ffb4ae",
        "body.help": "#e2c8ff",
        "attachment": "#7d8590 italic",
        "composer": "bg:#111821 #f0f6fc",
        "composer.prompt": "#7dd3fc bold",
        "composer.context": "bg:#0f151c #a7b7c9",
        "composer.shortcuts": "bg:#0f151c #768390",
        "statusbar": "bg:#0f151c #c9d1d9",
        "statusbar.meta": "bg:#0f151c #8b949e",
        "statusbar.state": "bg:#0f151c #f0f6fc bold",
    }
)


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RepoContext:
    """Stable repo information shown in the shell chrome."""

    cwd: Path
    cwd_label: str
    branch: str | None


# ── ChatShell ─────────────────────────────────────────────────────────────────


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
        self._message_entries: list[_TranscriptEntry] = []
        self._notice_entries: list[_TranscriptEntry] = []
        self._pending_assistant = ""
        self._submitted_inputs: list[str] = []
        self._history_index = 0
        self._lock = threading.Lock()
        self._state = "idle"
        self._state_detail = "Ready"
        self._submit_callback: Callable[[str], None] | None = None
        self._cancel_callback: Callable[[], None] | None = None

        self._history = InMemoryHistory()
        self._buffer = Buffer(
            multiline=False,
            completer=ThreadedCompleter(_ChatCompleter(command_names())),
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
            height=1,
            style="class:composer",
        )

        container = FloatContainer(
            content=HSplit(
                [
                    self._header_window,
                    self._transcript_window,
                    Window(height=1, char=" ", style="class:surface"),
                    self._input_window,
                    self._meta_row,
                    self._footer,
                ]
            ),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=10),
                ),
            ],
        )

        self._application = Application(
            layout=Layout(container, focused_element=self._input_window),
            key_bindings=self._build_key_bindings(),
            style=_STYLE,
            mouse_support=True,
            full_screen=False,
        )
        self.sync_session(session, clear_notices=True)

    def run(
        self,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], None],
    ) -> None:
        self._submit_callback = on_submit
        self._cancel_callback = on_cancel
        self.show_status("Ready. Use /help for commands and @ to attach files.")
        self._application.run()

    def request_exit(self) -> None:
        self._application.exit()

    def sync_session(self, session: ChatSession, *, clear_notices: bool = False) -> None:
        with self._lock:
            self._session = session
            self._message_entries = [_entry_from_message(message) for message in session.messages]
            self._pending_assistant = ""
            if clear_notices:
                self._notice_entries.clear()
        self._invalidate()

    def set_state(self, state: str, detail: str) -> None:
        with self._lock:
            self._state = state
            self._state_detail = detail
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
            lines.append(f"system preview: {_truncate(system_prompt, 100)}")
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
            lines.append(f"{role}: {_truncate(message.content, 120)}")
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
            self._notice_entries.append(_TranscriptEntry(kind=kind, title=title, body=body))
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

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event) -> None:
            buffer = self._buffer
            if (
                buffer.complete_state is not None
                and buffer.complete_state.current_completion is not None
            ):
                buffer.apply_completion(buffer.complete_state.current_completion)
                buffer.complete_state = None
                return
            self._submit_buffer()

        @kb.add("tab")
        def _next_completion(event) -> None:
            if self._buffer.complete_state is None:
                self._buffer.start_completion(select_first=False)
                return
            self._buffer.complete_next()

        @kb.add("s-tab")
        def _previous_completion(event) -> None:
            if self._buffer.complete_state is None:
                self._buffer.start_completion(select_first=False)
                return
            self._buffer.complete_previous()

        @kb.add("escape")
        def _escape(event) -> None:
            if self._state == "generating" and self._cancel_callback is not None:
                self._cancel_callback()
                return
            self._buffer.complete_state = None
            self._invalidate()

        @kb.add("c-c")
        def _interrupt(event) -> None:
            if self._state == "generating" and self._cancel_callback is not None:
                self._cancel_callback()
                return
            event.app.exit()

        @kb.add("c-l")
        def _redraw(event) -> None:
            event.app.renderer.clear()
            event.app.invalidate()

        @kb.add("up")
        def _history_up(event) -> None:
            self._history_previous()

        @kb.add("down")
        def _history_down(event) -> None:
            self._history_next()

        return kb

    def _header_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            title = self._session.title
            system_on = "system on" if self._session.system_message() else "system off"
            turns = sum(1 for msg in self._session.messages if msg.role == "user")

        model = _short_model_name(self._model_id)
        branch = self._repo.branch or "-"
        return [
            ("class:header.brand", " MLXs "),
            ("class:header.model", f"  {model}  "),
            ("class:header.path", f"  {self._repo.cwd_label}\n"),
            ("class:header.badge", f" {branch} "),
            ("class:header.meta", f"  {title}  "),
            ("class:header.meta", f"{turns} turns  "),
            ("class:header.badge", f" {system_on} "),
        ]

    def _composer_context_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            text = self._buffer.text
            state = self._state
            title = self._session.title
            system_prompt = self._session.system_message()

        if state == "generating":
            return [("class:composer.context", " assistant is responding live")]
        if state == "cancelling":
            return [("class:composer.context", " stopping the current turn cleanly")]

        stripped = text.strip()
        if not stripped:
            context = (
                f" working in {title} · system {'on' if system_prompt else 'off'}"
                " · ask anything or attach files with @path"
            )
            return [("class:composer.context", context)]

        if stripped == "?":
            return [
                (
                    "class:composer.context",
                    " help shortcut ready · press Enter to open commands and keys",
                )
            ]

        if stripped.startswith("/"):
            context = _command_context(stripped)
            return [("class:composer.context", f" {context}")]

        mentions = parse_file_mentions(text)
        if mentions:
            preview = ", ".join(match.path for match in mentions[:3])
            extra = f" +{len(mentions) - 3}" if len(mentions) > 3 else ""
            return [
                (
                    "class:composer.context",
                    f" {len(mentions)} file reference(s) ready · {preview}{extra}",
                )
            ]

        return [("class:composer.context", f" drafting a request in {title}")]

    def _composer_shortcuts_fragments(self) -> list[tuple[str, str]]:
        if self._state == "generating":
            text = "Esc cancel · Ctrl+C stop"
        elif self._state == "cancelling":
            text = "Waiting for generation to stop..."
        else:
            text = "/ commands · @ files · Tab complete"
        return [("class:composer.shortcuts", f" {text}")]

    def _footer_left_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            turns = sum(1 for msg in self._session.messages if msg.role == "user")
            system_on = self._session.system_message() is not None
        left = (
            f" {_short_model_name(self._model_id)}"
            f" · {self._repo.branch or '-'}"
            f" · {turns} turns"
            f" · system {'on' if system_on else 'off'}"
            f" · max {self._max_tokens}"
            f" · temp {self._temperature}"
        )
        return [("class:statusbar", left)]

    def _footer_right_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            state = self._state
            detail = self._state_detail
        state_label = {
            "idle": "Ready",
            "generating": "Streaming",
            "cancelling": "Stopping",
            "error": "Attention",
        }.get(state, state.replace("_", " ").title())
        return [
            ("class:statusbar.meta", detail + "  "),
            ("class:statusbar.state", state_label),
        ]

    def _transcript_fragments(self) -> list[tuple[str, str]]:
        with self._lock:
            entries = [*self._message_entries, *self._notice_entries]
            pending = self._pending_assistant

        fragments: list[tuple[str, str]] = []
        for index, entry in enumerate(entries):
            if index:
                fragments.append(("", "\n"))
            fragments.extend(_render_entry(entry))

        if pending:
            if entries:
                fragments.append(("", "\n"))
            fragments.extend(
                _render_entry(
                    _TranscriptEntry(kind="assistant", title="assistant", body=pending),
                    pending=True,
                )
            )

        if not fragments:
            fragments.extend(_empty_state_fragments())
        return fragments

    def _transcript_cursor(self) -> Point:
        text = "".join(part for _, part in self._transcript_fragments())
        lines = text.splitlines() or [""]
        return Point(x=len(lines[-1]), y=len(lines) - 1)

    def _invalidate(self) -> None:
        try:
            self._application.invalidate()
        except Exception:
            return


# ── Module-level helpers ──────────────────────────────────────────────────────


def parse_command(line: str) -> tuple[str, str] | None:
    """Parse an interactive slash command.

    Returns ``(name, args)`` or ``None`` if the line does not start with ``/``.
    """
    if not line.startswith("/"):
        return None
    body = line[1:].strip()
    if not body:
        return "", ""
    name, _, rest = body.partition(" ")
    return name.lower(), rest.strip()


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
        cwd_label=_display_path(workdir),
        branch=branch,
    )


# ── Private utilities ─────────────────────────────────────────────────────────


def _display_path(path: Path) -> str:
    home = Path.home()
    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    if not relative.parts:
        return "~"
    return f"~/{relative}"


def _short_model_name(model_id: str) -> str:
    if not model_id:
        return "default"
    return model_id.split("/")[-1]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
