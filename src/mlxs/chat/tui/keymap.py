"""Prompt-toolkit key bindings for the chat TUI."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.key_binding import KeyBindings


def build_chat_key_bindings(
    *,
    buffer,
    get_state: Callable[[], str],
    get_cancel_callback: Callable[[], Callable[[], None] | None],
    is_filter_focused: Callable[[], bool],
    is_palette_open: Callable[[], bool],
    is_palette_focused: Callable[[], bool],
    is_reference_picker_open: Callable[[], bool],
    is_reference_picker_focused: Callable[[], bool],
    is_help_open: Callable[[], bool],
    is_confirmation_open: Callable[[], bool],
    focus_filter: Callable[[], None],
    focus_composer: Callable[[], None],
    focus_palette: Callable[[], None],
    close_palette: Callable[[], None],
    open_reference_picker: Callable[[], None],
    close_reference_picker: Callable[[], None],
    close_help: Callable[[], None],
    confirm_accept: Callable[[], None],
    confirm_cancel: Callable[[], None],
    get_previous_session_callback: Callable[[], Callable[[], None] | None],
    get_next_session_callback: Callable[[], Callable[[], None] | None],
    palette_previous: Callable[[], None],
    palette_next: Callable[[], None],
    palette_accept: Callable[[], None],
    reference_previous: Callable[[], None],
    reference_next: Callable[[], None],
    reference_accept: Callable[[], None],
    submit_buffer: Callable[[], None],
    history_previous: Callable[[], None],
    history_next: Callable[[], None],
    invalidate: Callable[[], None],
) -> KeyBindings:
    """Build the key map used by the prompt-toolkit chat shell."""
    kb = KeyBindings()

    @kb.add("/")
    def _slash(event) -> None:
        if is_help_open():
            return
        if is_confirmation_open():
            return
        if (
            is_palette_focused()
            or is_reference_picker_focused()
            or is_filter_focused()
            or event.app.current_buffer is not buffer
        ):
            event.app.current_buffer.insert_text("/")
            return
        if get_state() == "idle" and not is_palette_open() and buffer.text == "":
            focus_palette()
            return
        buffer.insert_text("/")

    @kb.add("@")
    def _at(event) -> None:
        if is_help_open():
            return
        if is_confirmation_open():
            return
        if (
            is_palette_focused()
            or is_reference_picker_focused()
            or is_filter_focused()
            or event.app.current_buffer is not buffer
        ):
            event.app.current_buffer.insert_text("@")
            return
        if get_state() == "idle" and not is_palette_open() and not is_reference_picker_open():
            open_reference_picker()
            return
        buffer.insert_text("@")

    @kb.add("enter")
    def _newline(event) -> None:
        if is_help_open():
            return
        if is_confirmation_open():
            confirm_accept()
            return
        if is_reference_picker_focused():
            reference_accept()
            return
        if is_palette_focused():
            palette_accept()
            return
        if is_filter_focused():
            focus_composer()
            return
        if buffer.complete_state is not None and buffer.complete_state.current_completion is not None:
            buffer.apply_completion(buffer.complete_state.current_completion)
            buffer.complete_state = None
            return
        buffer.insert_text("\n")

    @kb.add("c-j")
    def _submit(event) -> None:
        if is_help_open():
            return
        if is_confirmation_open():
            confirm_accept()
            return
        if is_reference_picker_focused():
            reference_accept()
            return
        if is_palette_focused():
            palette_accept()
            return
        if is_filter_focused():
            focus_composer()
            return
        if buffer.complete_state is not None and buffer.complete_state.current_completion is not None:
            buffer.apply_completion(buffer.complete_state.current_completion)
            buffer.complete_state = None
            return
        submit_buffer()

    @kb.add("tab")
    def _next_completion(event) -> None:
        if is_help_open() or is_confirmation_open():
            return
        if buffer.complete_state is None:
            buffer.start_completion(select_first=False)
            return
        buffer.complete_next()

    @kb.add("s-tab")
    def _previous_completion(event) -> None:
        if is_help_open() or is_confirmation_open():
            return
        if buffer.complete_state is None:
            buffer.start_completion(select_first=False)
            return
        buffer.complete_previous()

    @kb.add("escape")
    def _escape(event) -> None:
        if is_confirmation_open():
            confirm_cancel()
            return
        if is_help_open():
            close_help()
            return
        if is_reference_picker_open():
            close_reference_picker()
            return
        if is_palette_open():
            close_palette()
            return
        cancel_callback = get_cancel_callback()
        if get_state() == "generating" and cancel_callback is not None:
            cancel_callback()
            return
        if is_filter_focused():
            focus_composer()
            return
        buffer.complete_state = None
        invalidate()

    @kb.add("c-c")
    def _interrupt(event) -> None:
        cancel_callback = get_cancel_callback()
        if get_state() == "generating" and cancel_callback is not None:
            cancel_callback()
            return
        event.app.exit()

    @kb.add("c-l")
    def _redraw(event) -> None:
        event.app.renderer.clear()
        event.app.invalidate()

    @kb.add("c-f")
    def _focus_filter(event) -> None:
        if get_state() != "idle":
            return
        if is_help_open():
            return
        if is_confirmation_open():
            return
        if is_reference_picker_open():
            focus_composer()
            return
        if is_palette_open():
            focus_palette()
            return
        focus_filter()

    @kb.add("up")
    def _history_up(event) -> None:
        if is_help_open():
            return
        if is_confirmation_open():
            return
        if is_reference_picker_focused():
            reference_previous()
            return
        if is_palette_focused():
            palette_previous()
            return
        if is_filter_focused():
            return
        history_previous()

    @kb.add("down")
    def _history_down(event) -> None:
        if is_help_open():
            return
        if is_confirmation_open():
            return
        if is_reference_picker_focused():
            reference_next()
            return
        if is_palette_focused():
            palette_next()
            return
        if is_filter_focused():
            return
        history_next()

    @kb.add("c-up")
    def _previous_session(event) -> None:
        if get_state() != "idle":
            return
        if is_help_open():
            return
        if is_confirmation_open():
            return
        callback = get_previous_session_callback()
        if callback is not None:
            callback()

    @kb.add("c-down")
    def _next_session(event) -> None:
        if get_state() != "idle":
            return
        if is_help_open():
            return
        if is_confirmation_open():
            return
        callback = get_next_session_callback()
        if callback is not None:
            callback()

    return kb


__all__ = ["build_chat_key_bindings"]
