"""Pure presentation helpers for the compact right context/status rail."""

from __future__ import annotations

from dataclasses import dataclass

from mlxs.chat.present.chrome import short_model_name, truncate_text
from mlxs.chat.session import ChatSession


@dataclass(frozen=True, slots=True)
class ContextRailRow:
    """One compact label/value row in the right context rail."""

    label: str
    value: str
    subdued: bool = False


@dataclass(frozen=True, slots=True)
class ContextRailSummary:
    """Presentation summary for the passive right context rail."""

    rows: tuple[ContextRailRow, ...]


def build_context_rail_summary(
    *,
    session: ChatSession,
    model_id: str,
    state: str,
    detail: str,
) -> ContextRailSummary:
    """Build a compact right-rail summary from real existing shell state."""
    state_label = {
        "idle": "Ready",
        "generating": "Running",
        "cancelling": "Cancelling",
        "error": "Attention",
    }.get(state, state.replace("_", " ").title())
    rows = [
        ContextRailRow("chat", truncate_text(session.title.strip() or "New chat", 22)),
        ContextRailRow("session", session.session_id),
        ContextRailRow("messages", str(len(session.messages))),
        ContextRailRow("system", "On" if session.system_message() else "Off"),
        ContextRailRow("model", truncate_text(short_model_name(model_id), 22)),
        ContextRailRow("state", state_label),
    ]
    detail_text = detail.strip()
    if detail_text and detail_text != state_label:
        rows.append(ContextRailRow("detail", truncate_text(detail_text, 22), subdued=True))
    return ContextRailSummary(rows=tuple(rows))


def empty_context_rail_fragments() -> list[tuple[str, str]]:
    """Render a minimal empty state for the right context rail."""
    return [
        ("class:context.title", " Context "),
        ("", "\n"),
        ("class:context.meta", " no active session"),
    ]


def render_context_rail_fragments(summary: ContextRailSummary | None) -> list[tuple[str, str]]:
    """Render the right context rail summary to prompt_toolkit fragments."""
    if summary is None:
        return empty_context_rail_fragments()

    fragments: list[tuple[str, str]] = [("class:context.title", " Context ")]
    for row in summary.rows:
        value_style = "class:context.meta" if row.subdued else "class:context.value"
        fragments.append(("", "\n"))
        fragments.append(("class:context.label", f" {row.label:<8}"))
        fragments.append((value_style, f" {row.value}"))
    return fragments


__all__ = [
    "ContextRailRow",
    "ContextRailSummary",
    "build_context_rail_summary",
    "empty_context_rail_fragments",
    "render_context_rail_fragments",
]
