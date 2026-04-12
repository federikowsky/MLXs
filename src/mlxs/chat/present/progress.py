"""Pure presentation helpers for the narrow progress/status strip."""

from __future__ import annotations


def progress_visible(*, state: str) -> bool:
    """Return whether the progress strip should be shown for the current state."""
    return state != "idle"


def build_progress_fragments(*, state: str, detail: str) -> list[tuple[str, str]]:
    """Build a restrained progress/status strip from real shell state only."""
    state_label = {
        "generating": "Running",
        "cancelling": "Cancelling",
        "error": "Attention",
    }.get(state, state.replace("_", " ").title())
    return [
        ("class:progress.state", f" {state_label} "),
        ("class:progress.meta", f" {detail}"),
    ]


__all__ = ["build_progress_fragments", "progress_visible"]
