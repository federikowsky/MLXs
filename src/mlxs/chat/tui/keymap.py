"""Prompt-toolkit key bindings for the chat TUI."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.key_binding import KeyBindings


def build_chat_key_bindings(
    *,
    buffer,
    get_state: Callable[[], str],
    get_cancel_callback: Callable[[], Callable[[], None] | None],
    submit_buffer: Callable[[], None],
    history_previous: Callable[[], None],
    history_next: Callable[[], None],
    invalidate: Callable[[], None],
) -> KeyBindings:
    """Build the key map used by the prompt-toolkit chat shell."""
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        if buffer.complete_state is not None and buffer.complete_state.current_completion is not None:
            buffer.apply_completion(buffer.complete_state.current_completion)
            buffer.complete_state = None
            return
        submit_buffer()

    @kb.add("tab")
    def _next_completion(event) -> None:
        if buffer.complete_state is None:
            buffer.start_completion(select_first=False)
            return
        buffer.complete_next()

    @kb.add("s-tab")
    def _previous_completion(event) -> None:
        if buffer.complete_state is None:
            buffer.start_completion(select_first=False)
            return
        buffer.complete_previous()

    @kb.add("escape")
    def _escape(event) -> None:
        cancel_callback = get_cancel_callback()
        if get_state() == "generating" and cancel_callback is not None:
            cancel_callback()
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

    @kb.add("up")
    def _history_up(event) -> None:
        history_previous()

    @kb.add("down")
    def _history_down(event) -> None:
        history_next()

    return kb


__all__ = ["build_chat_key_bindings"]
