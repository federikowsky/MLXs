"""Prompt-toolkit shell, command helpers, and attachment parsing for MLXs chat."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, ThreadedCompleter
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

from mlxs.chat.session import ChatMessage, ChatSession


@dataclass(frozen=True)
class CommandSpec:
    """Declarative description for a slash command."""

    name: str
    usage: str
    description: str


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


@dataclass(frozen=True)
class RepoContext:
    """Stable repo information shown in the shell chrome."""

    cwd: Path
    cwd_label: str
    branch: str | None


@dataclass(slots=True)
class _TranscriptEntry:
    kind: str
    title: str
    body: str
    attachments: tuple[str, ...] = ()


_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("help", "/help", "Show commands and keyboard shortcuts."),
    CommandSpec("model", "/model", "Show the active model and runtime settings."),
    CommandSpec("new", "/new", "Start a fresh chat session."),
    CommandSpec("clear", "/clear", "Clear the current conversation."),
    CommandSpec("undo", "/undo", "Remove the last user/assistant turn."),
    CommandSpec("retry", "/retry", "Regenerate the last user turn."),
    CommandSpec("system", "/system <text>|clear", "Set, inspect, or clear the system prompt."),
    CommandSpec("title", "/title <text>", "Rename the current chat session."),
    CommandSpec("history", "/history [n]", "Show recent transcript lines."),
    CommandSpec("export", "/export [path]", "Write the current transcript to Markdown."),
    CommandSpec("status", "/status", "Alias for /stats."),
    CommandSpec("stats", "/stats", "Show the current session summary."),
    CommandSpec("quit", "/quit", "Exit the chat."),
)

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


class AttachmentResolutionError(ValueError):
    """Raised when an `@file` mention cannot be attached."""


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
            completer=ThreadedCompleter(_ChatCompleter(_COMMANDS)),
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
        self._append_notice("help", "Commands", _help_card())

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
            return [
                (
                    "class:composer.context",
                    " assistant is responding live",
                )
            ]
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


class _ChatCompleter(Completer):
    """Completion menu for slash commands, /export paths, and @file mentions."""

    def __init__(self, command_specs: tuple[CommandSpec, ...]) -> None:
        self._command_specs = command_specs

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ):
        before = document.text_before_cursor
        stripped = before.lstrip()

        if stripped.startswith("/") and " " not in stripped[1:]:
            for spec in self._command_specs:
                candidate = f"/{spec.name}"
                if candidate.startswith(stripped):
                    yield Completion(
                        candidate,
                        start_position=-len(stripped),
                        display=candidate,
                        display_meta=spec.description,
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

        mention_token = _mention_completion_token(before)
        if mention_token is None:
            return
        mention_prefix = mention_token[1:]
        for candidate in _path_completion_values(mention_prefix, Path.cwd()):
            yield Completion(
                f"@{candidate}",
                start_position=-len(mention_token),
                display=f"@{candidate}",
                display_meta="attach file",
            )


def parse_command(line: str) -> tuple[str, str] | None:
    """Parse an interactive slash command."""
    if not line.startswith("/"):
        return None
    body = line[1:].strip()
    if not body:
        return "", ""
    name, _, rest = body.partition(" ")
    return name.lower(), rest.strip()


def command_names() -> tuple[str, ...]:
    """Return slash commands in the format expected by chat completers."""
    return tuple(f"/{spec.name}" for spec in _COMMANDS)


def help_card() -> str:
    """Return the help card text used by both interactive and plain chat."""
    return _help_card()


def build_completions(
    buffer: str,
    text: str,
    command_names: tuple[str, ...],
    *,
    cwd: Path | None = None,
) -> list[str]:
    """Pure completion helper used by unit tests."""
    workdir = cwd or Path.cwd()
    stripped = buffer.lstrip()
    if stripped.startswith("/") and " " not in stripped[1:]:
        return [name for name in command_names if name.startswith(text)]

    export_prefix = _export_completion_prefix(buffer)
    if export_prefix is not None:
        return _path_completion_values(export_prefix, workdir)

    mention_token = _mention_completion_token(buffer)
    if mention_token is not None:
        return [f"@{value}" for value in _path_completion_values(mention_token[1:], workdir)]

    return []


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


def _entry_from_message(message: ChatMessage) -> _TranscriptEntry:
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


def _render_entry(entry: _TranscriptEntry, *, pending: bool = False) -> list[tuple[str, str]]:
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


def _command_context(text: str) -> str:
    if text == "/":
        return "command mode · Tab shows the full command menu"
    matches = [spec for spec in _COMMANDS if f"/{spec.name}".startswith(text)]
    if not matches:
        return "unknown command · press Tab to inspect available commands"
    if len(matches) == 1 and f"/{matches[0].name}" == text:
        spec = matches[0]
        return f"{spec.usage} · {spec.description}"
    preview = " · ".join(f"{spec.usage} {spec.description}" for spec in matches[:3])
    if len(matches) > 3:
        preview += " · ..."
    return preview


def _help_card() -> str:
    width = max(len(spec.usage) for spec in _COMMANDS)
    lines = ["slash commands:"]
    lines.extend(f"  {spec.usage.ljust(width)}  {spec.description}" for spec in _COMMANDS)
    lines.extend(
        (
            "",
            "keyboard:",
            "  Enter          submit current input",
            "  Tab / Shift-Tab browse completion menu",
            "  Up / Down      recall previous inputs when composer is empty",
            "  Esc            cancel current generation",
            "  Ctrl+L         clear and redraw terminal",
            "  Ctrl+C         exit when idle, cancel when generating",
        )
    )
    return "\n".join(lines)


def _empty_state_fragments() -> list[tuple[str, str]]:
    lines = [
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
    return lines


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


def _export_completion_prefix(buffer: str) -> str | None:
    stripped = buffer.lstrip()
    if not stripped.startswith("/export "):
        return None
    return stripped[len("/export ") :]


def _mention_completion_token(buffer: str) -> str | None:
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
