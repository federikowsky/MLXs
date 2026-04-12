"""Tests for server chat loop — run_chat_loop with mocks (plan-chat-cli)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlxs.advanced_engines.prompt_cache import PromptCacheOrchestrator
from mlxs._types import FinishReason, GenerateOptions, TokenEvent
from mlxs.chat.session import ChatSession
from mlxs.chat.store import ChatSessionStore
from mlxs.config.schema import AppConfig
from mlxs.product_surfaces.chat import _handle_plain_command, run_chat_loop


def _make_deps(
    *,
    generate_events: list[TokenEvent],
    prompt_cache_get: tuple[list | None, int] = (None, 0),
    tokenizer_encode: list[int] | None = None,
    tokenizer_decode: str = "mock reply",
) -> SimpleNamespace:
    config = AppConfig()
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = tokenizer_encode or [1, 2, 3]
    tokenizer.decode.return_value = tokenizer_decode
    tokenizer.apply_chat_template = MagicMock(return_value="<|user|>hi<|assistant|>")
    tokenizer.eos_token_id = 2
    tokenizer.vocab_size = 1000

    prompt_cache = MagicMock()
    prompt_cache.get.return_value = prompt_cache_get
    prompt_cache.put = MagicMock()

    def generate_fn(*args, **kwargs):
        out = kwargs.get("final_cache_out")
        try:
            yield from generate_events
        finally:
            if out is not None:
                out.append([MagicMock()])

    return SimpleNamespace(
        config=config,
        model=model,
        tokenizer=tokenizer,
        prompt_cache=prompt_cache,
        prompt_cache_orchestrator=PromptCacheOrchestrator(prompt_cache),
        metrics=MagicMock(),
        generate_fn=generate_fn,
    )


class TestChatLoopHappyPath:
    """Single turn: input -> generate -> output -> put."""

    def test_messages_grow_and_prompt_cache_put_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = [
            TokenEvent(token_id=10, text="Hello"),
            TokenEvent(token_id=11, text=" world", finish_reason=FinishReason.STOP),
        ]
        deps = _make_deps(generate_events=events, tokenizer_encode=[1, 2, 3, 4])
        deps.tokenizer.encode.side_effect = [
            [1, 2, 3, 4],
            [9, 9, 9],
        ]
        lines_iter = iter(["hi", ""])

        def fake_read_line(prompt: str | None = None) -> str | None:
            assert prompt is None
            try:
                return next(lines_iter)
            except StopIteration:
                return None

        monkeypatch.setattr("mlxs.product_surfaces.chat._read_line", fake_read_line)
        run_chat_loop(deps)

        deps.tokenizer.apply_chat_template.assert_called()
        deps.prompt_cache.get.assert_called()
        deps.prompt_cache.put.assert_called_once()
        put_args = deps.prompt_cache.put.call_args
        assert put_args[0][0] == "default"
        assert put_args[0][1] == (9, 9, 9)

    def test_generate_fn_receives_final_cache_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            TokenEvent(token_id=1, text="x", finish_reason=FinishReason.STOP),
        ]
        deps = _make_deps(generate_events=events)
        deps.generate_fn = MagicMock(return_value=iter(events))
        call_count = 0

        def fake_read_line(prompt: str | None = None) -> str | None:
            assert prompt is None
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "hello"
            return None

        monkeypatch.setattr("mlxs.product_surfaces.chat._read_line", fake_read_line)
        run_chat_loop(deps)

        deps.generate_fn.assert_called()
        call_kw = deps.generate_fn.call_args.kwargs
        assert "final_cache_out" in call_kw
        assert call_kw["final_cache_out"] is not None


class TestChatLoopPromptCacheHit:
    """When prompt_cache.get returns a hit, suffix and cache passed to generate."""

    def test_cache_hit_uses_suffix_and_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_cache_state = [MagicMock()]
        events = [
            TokenEvent(token_id=99, text="cached", finish_reason=FinishReason.STOP),
        ]
        deps = _make_deps(
            generate_events=events,
            prompt_cache_get=(fake_cache_state, 2),
            tokenizer_encode=[1, 2, 3, 4, 5],
        )
        deps.generate_fn = MagicMock(return_value=iter(events))
        lines_iter = iter(["hi", None])

        def fake_read_line(prompt: str | None = None) -> str | None:
            assert prompt is None
            return next(lines_iter)

        monkeypatch.setattr("mlxs.product_surfaces.chat._read_line", fake_read_line)
        run_chat_loop(deps)

        deps.prompt_cache.get.assert_called_with("default", (1, 2, 3, 4, 5))
        deps.generate_fn.assert_called_once()
        call_args = deps.generate_fn.call_args
        prompt_arg = call_args[0][2]
        assert prompt_arg == [3, 4, 5]
        assert call_args.kwargs.get("cache") is fake_cache_state


class TestChatLoopKeyboardInterrupt:
    """KeyboardInterrupt exits cleanly without appending partial assistant."""

    def test_keyboard_interrupt_during_input_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        deps = _make_deps(generate_events=[])

        def raise_interrupt(prompt: str | None = None) -> str | None:
            assert prompt is None
            raise KeyboardInterrupt()

        monkeypatch.setattr("mlxs.product_surfaces.chat._read_line", raise_interrupt)
        run_chat_loop(deps)

        deps.prompt_cache.put.assert_not_called()

    def test_initial_query_runs_one_shot_without_reading_stdin(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        events = [
            TokenEvent(token_id=10, text="Hello"),
            TokenEvent(token_id=11, text=" world", finish_reason=FinishReason.STOP),
        ]
        deps = _make_deps(generate_events=events, tokenizer_encode=[1, 2, 3, 4])

        def fail_read_line(prompt: str | None = None) -> str | None:
            raise AssertionError("_read_line should not be called for one-shot chat")

        monkeypatch.setattr("mlxs.product_surfaces.chat._read_line", fail_read_line)
        run_chat_loop(deps, initial_query="hello")

        deps.prompt_cache.get.assert_called_once_with("default", (1, 2, 3, 4))
        deps.prompt_cache.put.assert_called_once()
        assert capsys.readouterr().out == "Hello world\n"

    def test_help_command_prints_commands_without_generating(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        deps = _make_deps(generate_events=[])
        deps.generate_fn = MagicMock(return_value=iter(()))
        lines_iter = iter(["/help", "/quit"])

        def fake_read_line(prompt: str | None = None) -> str | None:
            assert prompt is None
            return next(lines_iter)

        monkeypatch.setattr("mlxs.product_surfaces.chat._read_line", fake_read_line)
        run_chat_loop(deps)

        deps.generate_fn.assert_not_called()
        output = capsys.readouterr().out
        assert "Commands" in output
        assert "/retry" in output

    def test_export_command_writes_markdown_transcript(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        events = [
            TokenEvent(token_id=10, text="Hello", finish_reason=FinishReason.STOP),
        ]
        deps = _make_deps(generate_events=events, tokenizer_encode=[1, 2, 3, 4])
        export_path = tmp_path / "chat.md"
        lines_iter = iter(["hello", f"/export {export_path}", "/quit"])

        def fake_read_line(prompt: str | None = None) -> str | None:
            assert prompt is None
            return next(lines_iter)

        monkeypatch.setattr("mlxs.product_surfaces.chat._read_line", fake_read_line)
        run_chat_loop(deps)

        exported = export_path.read_text(encoding="utf-8")
        assert "# hello" in exported
        assert "## User" in exported
        assert "## Assistant" in exported

    def test_streamed_special_markers_are_not_printed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        events = [
            TokenEvent(token_id=10, text="Hello"),
            TokenEvent(token_id=11, text=" world<|im"),
            TokenEvent(token_id=12, text="_end|>", finish_reason=FinishReason.STOP),
        ]
        deps = _make_deps(
            generate_events=events,
            tokenizer_encode=[1, 2, 3, 4],
            tokenizer_decode="Hello world<|im_end|>",
        )

        def fail_read_line(prompt: str | None = None) -> str | None:
            raise AssertionError("_read_line should not be called for one-shot chat")

        monkeypatch.setattr("mlxs.product_surfaces.chat._read_line", fail_read_line)
        run_chat_loop(deps, initial_query="hello")

        assert capsys.readouterr().out == "Hello world\n"


class _RecordingConsole:
    def __init__(self) -> None:
        self.help_calls = 0
        self.statuses: list[str] = []
        self.errors: list[str] = []
        self.runtime_calls: list[tuple[int, float]] = []
        self.history_calls: list[int] = []
        self.stats_calls = 0

    def show_help(self) -> None:
        self.help_calls += 1

    def show_status(self, text: str) -> None:
        self.statuses.append(text)

    def show_error(self, text: str) -> None:
        self.errors.append(text)

    def show_runtime(self, *, max_tokens: int, temperature: float) -> None:
        self.runtime_calls.append((max_tokens, temperature))

    def show_history(self, session: ChatSession, limit: int) -> None:
        self.history_calls.append(limit)

    def show_stats(self, session: ChatSession) -> None:
        self.stats_calls += 1

    def stream_reply(self, text: str) -> None:
        return None

    def finish_reply(self, *, emitted_text: bool) -> None:
        return None

    def close_reply(self) -> None:
        return None


class TestPlainCommandActions:
    def _gen_opts(self) -> GenerateOptions:
        return GenerateOptions(max_tokens=512, temperature=1.0, stream=True)

    def _base_session(self) -> ChatSession:
        session = ChatSession(model_path="default")
        session.add_user_message("hello")
        session.add_assistant_message("world")
        return session

    def test_quit_and_exit_commands_request_exit(self) -> None:
        deps = _make_deps(generate_events=[])
        console = _RecordingConsole()
        session = ChatSession(model_path="default")

        for name in ("quit", "exit"):
            _, should_exit = _handle_plain_command(
                (name, ""),
                deps,
                session,
                self._gen_opts(),
                "default",
                console,
            )
            assert should_exit is True

    def test_help_and_model_commands_render_without_generation(self) -> None:
        deps = _make_deps(generate_events=[])
        console = _RecordingConsole()
        session = ChatSession(model_path="default")

        session, should_exit = _handle_plain_command(
            ("help", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert should_exit is False
        assert console.help_calls == 1

        _handle_plain_command(
            ("model", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert console.runtime_calls == [
            (deps.config.generate.max_tokens, deps.config.generate.temperature)
        ]

    def test_new_clear_undo_title_and_stats_commands_update_session(self) -> None:
        deps = _make_deps(generate_events=[])
        console = _RecordingConsole()
        session = self._base_session()
        session.set_system_message("system")

        session, _ = _handle_plain_command(
            ("undo", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert session.messages[0].role == "system"
        assert len(session.messages) == 1

        session.add_user_message("second")
        session.add_assistant_message("reply")
        session, _ = _handle_plain_command(
            ("clear", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert [message.role for message in session.messages] == ["system"]

        session, _ = _handle_plain_command(
            ("title", "Renamed"),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert session.title == "Renamed"

        _handle_plain_command(
            ("stats", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        _handle_plain_command(
            ("status", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert console.stats_calls == 2

        new_session, _ = _handle_plain_command(
            ("new", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert new_session is not session
        assert new_session.messages == []

    def test_new_command_updates_store_active_session(self) -> None:
        deps = _make_deps(generate_events=[])
        console = _RecordingConsole()
        store = ChatSessionStore()
        session = store.create_session(model_path="default")

        new_session, should_exit = _handle_plain_command(
            ("new", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
            store=store,
        )

        assert should_exit is False
        assert store.active_session_id == new_session.session_id
        assert store.get_active_session() is new_session
        assert len(store.list_summaries()) == 2

    def test_system_history_export_and_unknown_commands(self, tmp_path) -> None:
        deps = _make_deps(generate_events=[])
        console = _RecordingConsole()
        session = self._base_session()

        session, _ = _handle_plain_command(
            ("system", "You are concise"),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert session.system_message() == "You are concise"

        _handle_plain_command(
            ("system", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert "System prompt: You are concise" in console.statuses[-1]

        session, _ = _handle_plain_command(
            ("system", "clear"),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert session.system_message() is None

        _handle_plain_command(
            ("history", "2"),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert console.history_calls == [2]

        _handle_plain_command(
            ("history", "bad"),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert console.errors[-1] == "Usage: /history [n]"

        export_path = tmp_path / "out.md"
        _handle_plain_command(
            ("export", str(export_path)),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert export_path.exists()

        _handle_plain_command(
            ("unknown", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )
        assert console.errors[-1] == "Unknown command: /unknown. Use /help."

    def test_retry_command_replays_last_user_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        deps = _make_deps(generate_events=[])
        console = _RecordingConsole()
        session = ChatSession(model_path="default")
        session.add_user_message("hello")
        session.add_assistant_message("world")
        called: list[str] = []

        def fake_run_turn(*args, **kwargs) -> str:
            called.append(args[1].messages[-1].content)
            return "ok"

        monkeypatch.setattr("mlxs.product_surfaces.chat._run_turn", fake_run_turn)

        session, should_exit = _handle_plain_command(
            ("retry", ""),
            deps,
            session,
            self._gen_opts(),
            "default",
            console,
        )

        assert should_exit is False
        assert called == ["hello"]
        assert session.messages[-1].role == "user"
