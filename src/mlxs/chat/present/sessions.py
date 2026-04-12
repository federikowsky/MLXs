"""Pure presentation helpers for session/conversation summaries."""

from __future__ import annotations

from dataclasses import dataclass

from mlxs.chat.store import ChatSessionSummary


@dataclass(frozen=True, slots=True)
class SessionListItem:
    """Presentation-oriented view model for one session summary."""

    session_id: str
    title: str
    meta: str
    active: bool
    search_text: str


def item_from_summary(summary: ChatSessionSummary) -> SessionListItem:
    """Convert a store summary into a presentation view model."""
    title = summary.title.strip() or "New chat"
    message_label = "message" if summary.message_count == 1 else "messages"
    meta = f"{summary.message_count} {message_label} · {summary.updated_at}"
    search_text = " ".join(
        part.lower()
        for part in (summary.session_id, title, summary.model_path, summary.updated_at)
        if part
    )
    return SessionListItem(
        session_id=summary.session_id,
        title=title,
        meta=meta,
        active=summary.active,
        search_text=search_text,
    )


def items_from_summaries(
    summaries: list[ChatSessionSummary],
    *,
    query: str = "",
) -> list[SessionListItem]:
    """Map and optionally filter session summaries for future list UIs."""
    items = [item_from_summary(summary) for summary in summaries]
    text = query.strip().lower()
    if not text:
        return items
    return [item for item in items if text in item.search_text]


def empty_session_list_fragments() -> list[tuple[str, str]]:
    """Fragments for a future empty conversation list state."""
    return [
        ("class:label.status", " sessions "),
        ("class:body.status", "  No conversations yet.\n"),
        ("class:body.status", "  Start a chat to create the first session."),
    ]


__all__ = [
    "SessionListItem",
    "empty_session_list_fragments",
    "item_from_summary",
    "items_from_summaries",
]
