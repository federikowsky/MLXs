from __future__ import annotations

from types import SimpleNamespace

from prompt_toolkit.buffer import Buffer

from mlxs.chat.tui.keymap import build_chat_key_bindings


def test_build_chat_key_bindings_registers_expected_keys() -> None:
    kb = build_chat_key_bindings(
        buffer=Buffer(),
        get_state=lambda: "idle",
        get_cancel_callback=lambda: None,
        is_filter_focused=lambda: False,
        focus_filter=lambda: None,
        focus_composer=lambda: None,
        get_previous_session_callback=lambda: None,
        get_next_session_callback=lambda: None,
        submit_buffer=lambda: None,
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    bindings = {tuple(str(key) for key in binding.keys) for binding in kb.bindings}
    assert ("Keys.ControlM",) in bindings
    assert ("Keys.ControlJ",) in bindings
    assert ("Keys.ControlI",) in bindings
    assert ("Keys.BackTab",) in bindings
    assert ("Keys.Escape",) in bindings
    assert ("Keys.ControlC",) in bindings
    assert ("Keys.ControlF",) in bindings
    assert ("Keys.ControlL",) in bindings
    assert ("Keys.ControlUp",) in bindings
    assert ("Keys.ControlDown",) in bindings
    assert ("Keys.Up",) in bindings
    assert ("Keys.Down",) in bindings


def test_escape_invokes_cancel_when_generating() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = build_chat_key_bindings(
        buffer=buffer,
        get_state=lambda: "generating",
        get_cancel_callback=lambda: lambda: called.append("cancel"),
        is_filter_focused=lambda: False,
        focus_filter=lambda: None,
        focus_composer=lambda: None,
        get_previous_session_callback=lambda: None,
        get_next_session_callback=lambda: None,
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


def test_enter_inserts_newline_when_idle() -> None:
    buffer = Buffer()
    kb = build_chat_key_bindings(
        buffer=buffer,
        get_state=lambda: "idle",
        get_cancel_callback=lambda: None,
        is_filter_focused=lambda: False,
        focus_filter=lambda: None,
        focus_composer=lambda: None,
        get_previous_session_callback=lambda: None,
        get_next_session_callback=lambda: None,
        submit_buffer=lambda: None,
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    enter_binding = next(
        binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == ("Keys.ControlM",)
    )
    enter_binding.handler(SimpleNamespace(app=None))

    assert buffer.text == "\n"


def test_ctrl_j_submits_buffer() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = build_chat_key_bindings(
        buffer=buffer,
        get_state=lambda: "idle",
        get_cancel_callback=lambda: None,
        is_filter_focused=lambda: False,
        focus_filter=lambda: None,
        focus_composer=lambda: None,
        get_previous_session_callback=lambda: None,
        get_next_session_callback=lambda: None,
        submit_buffer=lambda: called.append("submit"),
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    submit_binding = next(
        binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == ("Keys.ControlJ",)
    )
    submit_binding.handler(SimpleNamespace(app=None))

    assert called == ["submit"]


def test_ctrl_up_and_ctrl_down_invoke_session_callbacks_only_when_idle() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = build_chat_key_bindings(
        buffer=buffer,
        get_state=lambda: "idle",
        get_cancel_callback=lambda: None,
        is_filter_focused=lambda: False,
        focus_filter=lambda: None,
        focus_composer=lambda: None,
        get_previous_session_callback=lambda: lambda: called.append("prev"),
        get_next_session_callback=lambda: lambda: called.append("next"),
        submit_buffer=lambda: None,
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    prev_binding = next(
        binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == ("Keys.ControlUp",)
    )
    next_binding = next(
        binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == ("Keys.ControlDown",)
    )
    prev_binding.handler(SimpleNamespace(app=None))
    next_binding.handler(SimpleNamespace(app=None))

    assert called == ["prev", "next"]


def test_ctrl_up_does_not_switch_sessions_while_generating() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = build_chat_key_bindings(
        buffer=buffer,
        get_state=lambda: "generating",
        get_cancel_callback=lambda: None,
        is_filter_focused=lambda: False,
        focus_filter=lambda: None,
        focus_composer=lambda: None,
        get_previous_session_callback=lambda: lambda: called.append("prev"),
        get_next_session_callback=lambda: lambda: called.append("next"),
        submit_buffer=lambda: None,
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    prev_binding = next(
        binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == ("Keys.ControlUp",)
    )
    prev_binding.handler(SimpleNamespace(app=None))

    assert called == []


def test_ctrl_f_focuses_filter_when_idle() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = build_chat_key_bindings(
        buffer=buffer,
        get_state=lambda: "idle",
        get_cancel_callback=lambda: None,
        is_filter_focused=lambda: False,
        focus_filter=lambda: called.append("focus-filter"),
        focus_composer=lambda: called.append("focus-composer"),
        get_previous_session_callback=lambda: None,
        get_next_session_callback=lambda: None,
        submit_buffer=lambda: None,
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    binding = next(
        binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == ("Keys.ControlF",)
    )
    binding.handler(SimpleNamespace(app=None))

    assert called == ["focus-filter"]


def test_escape_returns_focus_from_filter_to_composer() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = build_chat_key_bindings(
        buffer=buffer,
        get_state=lambda: "idle",
        get_cancel_callback=lambda: None,
        is_filter_focused=lambda: True,
        focus_filter=lambda: called.append("focus-filter"),
        focus_composer=lambda: called.append("focus-composer"),
        get_previous_session_callback=lambda: None,
        get_next_session_callback=lambda: None,
        submit_buffer=lambda: None,
        history_previous=lambda: None,
        history_next=lambda: None,
        invalidate=lambda: None,
    )

    binding = next(
        binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == ("Keys.Escape",)
    )
    binding.handler(SimpleNamespace(app=None))

    assert called == ["focus-composer"]
