"""Tests for server chat loop — run_chat_loop with mocks (plan-chat-cli)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlxs._types import FinishReason, TokenEvent
from mlxs.config.schema import AppConfig
from mlxs.server.chat import run_chat_loop
from mlxs.server.deps import Dependencies


def _make_deps(
    *,
    generate_events: list[TokenEvent],
    prompt_cache_get: tuple[list | None, int] = (None, 0),
    tokenizer_encode: list[int] | None = None,
    tokenizer_decode: str = "mock reply",
) -> Dependencies:
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

    return Dependencies(
        config=config,
        model=model,
        tokenizer=tokenizer,
        prompt_cache=prompt_cache,
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
        lines_iter = iter(["hi", ""])

        def fake_read_line() -> str | None:
            try:
                return next(lines_iter)
            except StopIteration:
                return None

        monkeypatch.setattr("mlxs.server.chat._read_line", fake_read_line)
        run_chat_loop(deps)

        deps.tokenizer.apply_chat_template.assert_called()
        deps.prompt_cache.get.assert_called()
        deps.prompt_cache.put.assert_called_once()
        put_args = deps.prompt_cache.put.call_args
        assert put_args[0][0] == "default"
        assert put_args[0][1] == (1, 2, 3, 4, 10, 11)

    def test_generate_fn_receives_final_cache_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            TokenEvent(token_id=1, text="x", finish_reason=FinishReason.STOP),
        ]
        deps = _make_deps(generate_events=events)
        deps.generate_fn = MagicMock(return_value=iter(events))
        call_count = 0

        def fake_read_line() -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "hello"
            return None

        monkeypatch.setattr("mlxs.server.chat._read_line", fake_read_line)
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

        def fake_read_line() -> str | None:
            return next(lines_iter)

        monkeypatch.setattr("mlxs.server.chat._read_line", fake_read_line)
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

        def raise_interrupt() -> str | None:
            raise KeyboardInterrupt()

        monkeypatch.setattr("mlxs.server.chat._read_line", raise_interrupt)
        run_chat_loop(deps)

        deps.prompt_cache.put.assert_not_called()
