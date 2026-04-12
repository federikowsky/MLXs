from __future__ import annotations

from types import SimpleNamespace

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from mlxs.chat.tui.keymap import build_chat_key_bindings


def _make_keymap(
    *,
    buffer: Buffer | None = None,
    get_state=lambda: "idle",
    get_cancel_callback=lambda: None,
    is_filter_focused=lambda: False,
    is_palette_open=lambda: False,
    is_palette_focused=lambda: False,
    is_reference_picker_open=lambda: False,
    is_reference_picker_focused=lambda: False,
    focus_filter=lambda: None,
    focus_composer=lambda: None,
    focus_palette=lambda: None,
    close_palette=lambda: None,
    open_reference_picker=lambda: None,
    close_reference_picker=lambda: None,
    get_previous_session_callback=lambda: None,
    get_next_session_callback=lambda: None,
    palette_previous=lambda: None,
    palette_next=lambda: None,
    palette_accept=lambda: None,
    reference_previous=lambda: None,
    reference_next=lambda: None,
    reference_accept=lambda: None,
    submit_buffer=lambda: None,
    history_previous=lambda: None,
    history_next=lambda: None,
    invalidate=lambda: None,
):
    return build_chat_key_bindings(
        buffer=buffer or Buffer(),
        get_state=get_state,
        get_cancel_callback=get_cancel_callback,
        is_filter_focused=is_filter_focused,
        is_palette_open=is_palette_open,
        is_palette_focused=is_palette_focused,
        is_reference_picker_open=is_reference_picker_open,
        is_reference_picker_focused=is_reference_picker_focused,
        focus_filter=focus_filter,
        focus_composer=focus_composer,
        focus_palette=focus_palette,
        close_palette=close_palette,
        open_reference_picker=open_reference_picker,
        close_reference_picker=close_reference_picker,
        get_previous_session_callback=get_previous_session_callback,
        get_next_session_callback=get_next_session_callback,
        palette_previous=palette_previous,
        palette_next=palette_next,
        palette_accept=palette_accept,
        reference_previous=reference_previous,
        reference_next=reference_next,
        reference_accept=reference_accept,
        submit_buffer=submit_buffer,
        history_previous=history_previous,
        history_next=history_next,
        invalidate=invalidate,
    )


def _binding(kb, key_tuple: tuple[str, ...]):
    return next(binding for binding in kb.bindings if tuple(str(key) for key in binding.keys) == key_tuple)


def test_build_chat_key_bindings_registers_expected_keys() -> None:
    kb = _make_keymap()

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
    assert ("/",) in bindings
    assert ("@",) in bindings


def test_escape_invokes_cancel_when_generating() -> None:
    called: list[str] = []
    kb = _make_keymap(
        get_state=lambda: "generating",
        get_cancel_callback=lambda: lambda: called.append("cancel"),
    )

    _binding(kb, ("Keys.Escape",)).handler(SimpleNamespace(app=None))

    assert called == ["cancel"]


def test_enter_inserts_newline_when_idle() -> None:
    buffer = Buffer()
    kb = _make_keymap(buffer=buffer)

    _binding(kb, ("Keys.ControlM",)).handler(SimpleNamespace(app=None))

    assert buffer.text == "\n"


def test_ctrl_j_submits_buffer() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = _make_keymap(buffer=buffer, submit_buffer=lambda: called.append("submit"))

    _binding(kb, ("Keys.ControlJ",)).handler(SimpleNamespace(app=None))

    assert called == ["submit"]


def test_ctrl_up_and_ctrl_down_invoke_session_callbacks_only_when_idle() -> None:
    called: list[str] = []
    kb = _make_keymap(
        get_previous_session_callback=lambda: lambda: called.append("prev"),
        get_next_session_callback=lambda: lambda: called.append("next"),
    )

    _binding(kb, ("Keys.ControlUp",)).handler(SimpleNamespace(app=None))
    _binding(kb, ("Keys.ControlDown",)).handler(SimpleNamespace(app=None))

    assert called == ["prev", "next"]


def test_ctrl_up_does_not_switch_sessions_while_generating() -> None:
    called: list[str] = []
    kb = _make_keymap(
        get_state=lambda: "generating",
        get_previous_session_callback=lambda: lambda: called.append("prev"),
    )

    _binding(kb, ("Keys.ControlUp",)).handler(SimpleNamespace(app=None))

    assert called == []


def test_ctrl_f_focuses_filter_when_idle() -> None:
    called: list[str] = []
    kb = _make_keymap(
        focus_filter=lambda: called.append("focus-filter"),
        focus_composer=lambda: called.append("focus-composer"),
        focus_palette=lambda: called.append("focus-palette"),
    )

    _binding(kb, ("Keys.ControlF",)).handler(SimpleNamespace(app=None))

    assert called == ["focus-filter"]


def test_ctrl_f_keeps_palette_focus_when_palette_is_open() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = _make_keymap(
        is_palette_open=lambda: True,
        is_palette_focused=lambda: True,
        focus_filter=lambda: called.append("focus-filter"),
        focus_composer=lambda: called.append("focus-composer"),
        focus_palette=lambda: called.append("focus-palette"),
        buffer=buffer,
    )

    _binding(kb, ("Keys.ControlF",)).handler(SimpleNamespace(app=SimpleNamespace(current_buffer=buffer)))

    assert called == ["focus-palette"]


def test_ctrl_f_closes_reference_picker_back_to_composer() -> None:
    called: list[str] = []
    kb = _make_keymap(
        is_reference_picker_open=lambda: True,
        focus_composer=lambda: called.append("focus-composer"),
    )

    _binding(kb, ("Keys.ControlF",)).handler(SimpleNamespace(app=None))

    assert called == ["focus-composer"]


def test_escape_returns_focus_from_filter_to_composer() -> None:
    called: list[str] = []
    kb = _make_keymap(
        is_filter_focused=lambda: True,
        focus_composer=lambda: called.append("focus-composer"),
    )

    _binding(kb, ("Keys.Escape",)).handler(SimpleNamespace(app=None))

    assert called == ["focus-composer"]


def test_escape_closes_reference_picker() -> None:
    called: list[str] = []
    kb = _make_keymap(
        is_reference_picker_open=lambda: True,
        close_reference_picker=lambda: called.append("close-reference"),
    )

    _binding(kb, ("Keys.Escape",)).handler(SimpleNamespace(app=None))

    assert called == ["close-reference"]


def test_slash_opens_palette_when_composer_is_empty() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = _make_keymap(
        buffer=buffer,
        focus_palette=lambda: called.append("focus-palette"),
        palette_accept=lambda: called.append("accept-palette"),
    )

    _binding(kb, ("/",)).handler(SimpleNamespace(app=SimpleNamespace(current_buffer=buffer)))

    assert called == ["focus-palette"]
    assert buffer.text == ""


def test_slash_inserts_text_when_composer_is_not_empty() -> None:
    buffer = Buffer()
    buffer.document = Document("he", cursor_position=2)
    kb = _make_keymap(buffer=buffer)

    _binding(kb, ("/",)).handler(SimpleNamespace(app=SimpleNamespace(current_buffer=buffer)))

    assert buffer.text == "he/"


def test_slash_inserts_into_filter_buffer_when_filter_is_focused() -> None:
    buffer = Buffer()
    filter_buffer = Buffer()
    kb = _make_keymap(buffer=buffer, is_filter_focused=lambda: True)

    _binding(kb, ("/",)).handler(SimpleNamespace(app=SimpleNamespace(current_buffer=filter_buffer)))

    assert buffer.text == ""
    assert filter_buffer.text == "/"


def test_at_opens_reference_picker_when_idle() -> None:
    buffer = Buffer()
    called: list[str] = []
    kb = _make_keymap(
        buffer=buffer,
        open_reference_picker=lambda: called.append("open-reference"),
    )

    _binding(kb, ("@",)).handler(SimpleNamespace(app=SimpleNamespace(current_buffer=buffer)))

    assert called == ["open-reference"]
    assert buffer.text == ""


def test_at_inserts_text_when_reference_picker_already_open() -> None:
    buffer = Buffer()
    picker_buffer = Buffer()
    kb = _make_keymap(
        buffer=buffer,
        is_reference_picker_open=lambda: True,
        is_reference_picker_focused=lambda: True,
    )

    _binding(kb, ("@",)).handler(SimpleNamespace(app=SimpleNamespace(current_buffer=picker_buffer)))

    assert picker_buffer.text == "@"
    assert buffer.text == ""


def test_at_delegates_to_reference_picker_handler_from_composer() -> None:
    buffer = Buffer(document=Document("email", cursor_position=5))
    called: list[str] = []
    kb = _make_keymap(
        buffer=buffer,
        open_reference_picker=lambda: called.append("open-reference"),
    )

    _binding(kb, ("@",)).handler(SimpleNamespace(app=SimpleNamespace(current_buffer=buffer)))

    assert called == ["open-reference"]
    assert buffer.text == "email"


def test_up_and_down_navigate_reference_picker_when_focused() -> None:
    called: list[str] = []
    kb = _make_keymap(
        is_reference_picker_focused=lambda: True,
        reference_previous=lambda: called.append("prev-ref"),
        reference_next=lambda: called.append("next-ref"),
    )

    _binding(kb, ("Keys.Up",)).handler(SimpleNamespace(app=None))
    _binding(kb, ("Keys.Down",)).handler(SimpleNamespace(app=None))

    assert called == ["prev-ref", "next-ref"]


def test_enter_accepts_reference_picker_when_focused() -> None:
    called: list[str] = []
    kb = _make_keymap(
        is_reference_picker_focused=lambda: True,
        reference_accept=lambda: called.append("accept-ref"),
    )

    _binding(kb, ("Keys.ControlM",)).handler(SimpleNamespace(app=None))

    assert called == ["accept-ref"]
