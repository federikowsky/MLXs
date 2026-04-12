from __future__ import annotations

from types import SimpleNamespace

from prompt_toolkit.buffer import Buffer

from mlxs.chat.tui.keymap import build_chat_key_bindings


def test_build_chat_key_bindings_registers_expected_keys() -> None:
    kb = build_chat_key_bindings(
        buffer=Buffer(),
        get_state=lambda: "idle",
        get_cancel_callback=lambda: None,
        submit_buffer=lambda: None,
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    bindings = {tuple(str(key) for key in binding.keys) for binding in kb.bindings}
    assert ("Keys.ControlM",) in bindings
    assert ("Keys.ControlI",) in bindings
    assert ("Keys.BackTab",) in bindings
    assert ("Keys.Escape",) in bindings
    assert ("Keys.ControlC",) in bindings
    assert ("Keys.ControlL",) in bindings
    assert ("Keys.Up",) in bindings
    assert ("Keys.Down",) in bindings


def test_escape_invokes_cancel_when_generating() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = build_chat_key_bindings(
        buffer=buffer,
        get_state=lambda: "generating",
        get_cancel_callback=lambda: lambda: called.append("cancel"),
        submit_buffer=lambda: None,
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    escape_binding = next(
        binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == ("Keys.Escape",)
    )
    escape_binding.handler(SimpleNamespace(app=None))

    assert called == ["cancel"]
