"""MLXs Chat TUI — main Textual application.

Launch with:  mlxs chat --model <path-or-hf-id>

Features:
- Streaming assistant responses with Markdown rendering
- Multiline input (Shift+Enter for newline, Enter to send)
- Slash commands (/new, /save, /load, /clear, /retry, /delete_last, /rename, /help, /quit)
- Keyboard shortcuts
- Session persistence
- Status bar with model info
"""

from __future__ import annotations

import logging
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header

from mlxs.chat.controller import ChatController, ChatEvent, EventKind
from mlxs.chat.store import SessionStore
from mlxs.server.deps import Dependencies
from mlxs.ui.tui.widgets.composer import Composer
from mlxs.ui.tui.widgets.status_bar import StatusBar
from mlxs.ui.tui.widgets.transcript import Transcript

logger = logging.getLogger(__name__)


class ChatApp(App):
    """MLXs interactive chat TUI."""

    TITLE = "MLXs Chat"
    CSS = """
    Screen {
        background: $surface;
    }
    #chat-view {
        height: 1fr;
        padding: 0;
    }
    """

    BINDINGS = [  # noqa: RUF012
        Binding("ctrl+n", "new_session", "New", show=True),
        Binding("ctrl+s", "save_session", "Save", show=True),
        Binding("ctrl+o", "load_session", "Open", show=True),
        Binding("ctrl+r", "retry", "Retry", show=True),
        Binding("ctrl+l", "clear_session", "Clear", show=True),
        Binding("ctrl+q", "quit_app", "Quit", show=True),
        Binding("escape", "cancel_gen", "Cancel", show=False),
    ]

    def __init__(self, deps: Dependencies, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._deps = deps
        self._store = SessionStore()
        self._controller: ChatController | None = None
        self._streaming_text: list[str] = []
        self._session_list_cache: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="chat-view"):
            yield Transcript(id="transcript")
        yield StatusBar(id="status-bar")
        yield Composer(id="composer")
        yield Footer()

    def on_mount(self) -> None:
        self._controller = ChatController(
            deps=self._deps,
            store=self._store,
            on_event=self._handle_controller_event,
        )
        # Set initial status
        status = self.query_one("#status-bar", StatusBar)
        model_short = (self._deps.config.model.model_path or "unknown").rsplit("/", 1)[-1]
        status.model_name = model_short
        status.temperature = self._controller.temperature
        status.max_tokens = self._controller.max_tokens
        status.status_text = "Ready"
        status.session_title = self._controller.session.title

        # Welcome message
        transcript = self.query_one("#transcript", Transcript)
        transcript.add_system_message(
            f"MLXs Chat — model: {model_short}  |  "
            "Type a message, or /help for commands. Ctrl+Q to quit."
        )

    # -- Message handling --------------------------------------------------

    def on_composer_submitted(self, event: Composer.Submitted) -> None:
        """Handle submitted text from the composer."""
        text = event.text
        if text.startswith("/"):
            self._handle_slash_command(text)
        else:
            self._send_user_message(text)

    def _handle_slash_command(self, text: str) -> None:
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        transcript = self.query_one("#transcript", Transcript)

        if cmd == "/new":
            self.action_new_session()
        elif cmd == "/save":
            self.action_save_session()
        elif cmd == "/load":
            if arg:
                self._load_session_by_arg(arg)
            else:
                self._show_session_list()
        elif cmd == "/clear":
            self.action_clear_session()
        elif cmd == "/retry":
            self.action_retry()
        elif cmd == "/delete_last":
            self._do_delete_last()
        elif cmd == "/rename":
            if arg:
                self._controller.rename_session(arg)
                self._update_status()
            else:
                transcript.add_error_message("Usage: /rename <new title>")
        elif cmd == "/help":
            self._show_help()
        elif cmd == "/quit":
            self.action_quit_app()
        else:
            transcript.add_error_message(f"Unknown command: {cmd}. Type /help for list.")

    @work(thread=True)
    def _send_user_message(self, text: str) -> None:
        """Send message in a worker thread so generation doesn't block the UI."""
        # Show user message in transcript from the main thread
        self.call_from_thread(self._show_user_message, text)
        self._controller.send_message(text)

    def _show_user_message(self, text: str) -> None:
        transcript = self.query_one("#transcript", Transcript)
        transcript.add_user_message(text)

    # -- Controller event handling -----------------------------------------

    def _handle_controller_event(self, event: ChatEvent) -> None:
        """Called from the worker thread — must use call_from_thread for UI updates."""
        if event.kind == EventKind.GENERATION_STARTED:
            self.call_from_thread(self._on_generation_started)
        elif event.kind == EventKind.TOKEN_CHUNK:
            self.call_from_thread(self._on_token_chunk, event.text)
        elif event.kind == EventKind.GENERATION_FINISHED:
            self.call_from_thread(self._on_generation_finished, event.text)
        elif event.kind == EventKind.ERROR:
            self.call_from_thread(self._on_error, event.text)
        elif event.kind == EventKind.STATUS:
            self.call_from_thread(self._on_status, event.text)

    def _on_generation_started(self) -> None:
        self._streaming_text.clear()
        status = self.query_one("#status-bar", StatusBar)
        status.status_text = "Generating..."
        transcript = self.query_one("#transcript", Transcript)
        transcript.start_assistant_message()

    def _on_token_chunk(self, text: str) -> None:
        self._streaming_text.append(text)
        transcript = self.query_one("#transcript", Transcript)
        transcript.append_token(text)

    def _on_generation_finished(self, full_text: str) -> None:
        transcript = self.query_one("#transcript", Transcript)
        transcript.finish_assistant_message(full_text)
        status = self.query_one("#status-bar", StatusBar)
        status.status_text = "Ready"
        self._streaming_text.clear()
        self._update_status()
        # Auto-focus composer
        self.query_one("#composer", Composer).focus()

    def _on_error(self, text: str) -> None:
        transcript = self.query_one("#transcript", Transcript)
        transcript.add_error_message(text)
        status = self.query_one("#status-bar", StatusBar)
        status.status_text = "Ready"
        self.query_one("#composer", Composer).focus()

    def _on_status(self, text: str) -> None:
        transcript = self.query_one("#transcript", Transcript)
        transcript.add_system_message(text)
        self._update_status()

    # -- Actions -----------------------------------------------------------

    def action_new_session(self) -> None:
        if self._controller:
            self._controller.new_session()
            self.query_one("#transcript", Transcript).clear_transcript()
            self._update_status()

    def action_save_session(self) -> None:
        if self._controller:
            self._controller.save_session()

    def action_load_session(self) -> None:
        self._show_session_list()

    def action_clear_session(self) -> None:
        if self._controller:
            self._controller.clear_session()
            self.query_one("#transcript", Transcript).clear_transcript()
            self._update_status()

    def action_retry(self) -> None:
        if self._controller and not self._controller.is_generating:
            self._do_retry()

    @work(thread=True)
    def _do_retry(self) -> None:
        # Pop the old assistant + user from session FIRST (in worker thread),
        # then sync the transcript via main thread BEFORE re-generating.
        session = self._controller.session
        if not session.messages:
            return
        if session.messages[-1].role == "assistant":
            session.messages.pop()
        if session.messages and session.messages[-1].role == "user":
            last_user = session.messages.pop()
            # Sync transcript to the now-modified session (main thread)
            self.call_from_thread(self._refresh_transcript)
            # Re-generate (this adds user msg back + generates assistant)
            self._controller.send_message(last_user.content)

    def action_cancel_gen(self) -> None:
        if self._controller and self._controller.is_generating:
            self._controller.cancel_generation()

    def action_quit_app(self) -> None:
        if self._controller:
            self._controller.save_session()
        self.exit()

    def _do_delete_last(self) -> None:
        if self._controller:
            self._controller.delete_last_turn()
            # Re-render transcript
            self.query_one("#transcript", Transcript).clear_transcript()
            self._replay_session_to_transcript()

    # -- Session list (simple text-based) ----------------------------------

    def _show_session_list(self) -> None:
        if not self._controller:
            return
        sessions = self._controller.list_sessions()
        transcript = self.query_one("#transcript", Transcript)
        if not sessions:
            transcript.add_system_message("No saved sessions.")
            return
        transcript.add_system_message("Saved sessions (use /load <number> or /load <id>):")
        for i, s in enumerate(sessions[:20], 1):
            sid = s["session_id"]
            title = s["title"]
            count = s["message_count"]
            transcript.add_system_message(f"  {i}. [{sid}] {title} ({count} msgs)")
        self._session_list_cache = sessions[:20]

    def _load_session_by_arg(self, arg: str) -> None:
        """Load a session by numeric index (from last /load list) or by session ID."""
        if not self._controller:
            return
        transcript = self.query_one("#transcript", Transcript)

        # Try numeric index first (1-based, from last shown list)
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(self._session_list_cache):
                session_id = self._session_list_cache[idx]["session_id"]
            else:
                transcript.add_error_message(
                    f"Invalid index: {arg}. Use /load to see the list."
                )
                return
        else:
            # Treat as direct session_id
            session_id = arg.strip()

        try:
            self._controller.load_session(session_id)
            # Refresh transcript with loaded session
            self.query_one("#transcript", Transcript).clear_transcript()
            self._replay_session_to_transcript()
            self._update_status()
        except (FileNotFoundError, KeyError):
            transcript.add_error_message(f"Session not found: {session_id}")
        except Exception as exc:
            transcript.add_error_message(f"Failed to load session: {exc}")

    def _show_help(self) -> None:
        transcript = self.query_one("#transcript", Transcript)
        transcript.add_system_message("Commands:")
        commands = [
            "/new          — New session",
            "/save         — Save current session",
            "/load         — List saved sessions",
            "/load <n|id>  — Load by number or session id",
            "/clear        — Clear current session",
            "/retry        — Retry last generation",
            "/delete_last  — Delete last exchange",
            "/rename <t>   — Rename session",
            "/help         — Show this help",
            "/quit         — Quit",
        ]
        for c in commands:
            transcript.add_system_message(f"  {c}")
        transcript.add_system_message("Shortcuts:")
        shortcuts = [
            "Enter         — Send message",
            "Shift+Enter   — New line",
            "Ctrl+N        — New session",
            "Ctrl+S        — Save session",
            "Ctrl+O        — Open session",
            "Ctrl+R        — Retry",
            "Ctrl+L        — Clear",
            "Ctrl+Q        — Quit",
            "Escape        — Cancel generation",
        ]
        for s in shortcuts:
            transcript.add_system_message(f"  {s}")

    # -- Helpers -----------------------------------------------------------

    def _update_status(self) -> None:
        if not self._controller:
            return
        status = self.query_one("#status-bar", StatusBar)
        status.session_title = self._controller.session.title

    def _replay_session_to_transcript(self) -> None:
        """Re-render all messages from the current session into the transcript."""
        if not self._controller:
            return
        transcript = self.query_one("#transcript", Transcript)
        for msg in self._controller.session.messages:
            if msg.role == "user":
                transcript.add_user_message(msg.content)
            elif msg.role == "assistant":
                transcript.finish_assistant_message(msg.content)
            elif msg.role == "system":
                transcript.add_system_message(msg.content)

    def _refresh_transcript(self) -> None:
        """Clear transcript and replay current session state."""
        self.query_one("#transcript", Transcript).clear_transcript()
        self._replay_session_to_transcript()


def run_tui(deps: Dependencies) -> None:
    """Entry point to launch the TUI chat app."""
    app = ChatApp(deps)
    app.run()
