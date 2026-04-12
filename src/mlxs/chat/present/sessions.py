"""Pure presentation helpers for session/conversation summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mlxs.chat.session import ChatSession
from mlxs.chat.store import ChatSessionSummary


@dataclass(frozen=True, slots=True)
class SessionListItem:
    """Presentation-oriented view model for one session summary."""

    session_id: str
    title: str
    meta: str
    active: bool
    search_text: str


def format_updated_label(value: str) -> str:
    """Format an ISO timestamp into a compact rail-friendly label."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.strftime("%Y-%m-%d %H:%M")


def item_from_session(session: ChatSession, *, active: bool = True) -> SessionListItem:
    """Convert a live session directly into a rail-ready item."""
    title = session.title.strip() or "New chat"
    message_count = len(session.messages)
    message_label = "message" if message_count == 1 else "messages"
    meta = f"{message_count} {message_label} · {format_updated_label(session.updated_at)}"
    search_text = " ".join(
        part.lower()
        for part in (session.session_id, title, session.model_path, session.updated_at)
        if part
    )
    return SessionListItem(
        session_id=session.session_id,
        title=title,
        meta=meta,
        active=active,
        search_text=search_text,
    )


def item_from_summary(summary: ChatSessionSummary) -> SessionListItem:
    """Convert a store summary into a presentation view model."""
    title = summary.title.strip() or "New chat"
    message_label = "message" if summary.message_count == 1 else "messages"
    meta = f"{summary.message_count} {message_label} · {format_updated_label(summary.updated_at)}"
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


def render_session_list_fragments(items: list[SessionListItem]) -> list[tuple[str, str]]:
    """Render session rail items to prompt-toolkit text fragments."""
    if not items:
        return empty_session_list_fragments()

    fragments: list[tuple[str, str]] = [("class:rail.title", " Conversations ")]
    for item in items:
        title_style = "class:rail.item.active" if item.active else "class:rail.item"
        prefix = "> " if item.active else "  "
        fragments.append(("", "\n"))
        fragments.append((title_style, f"{prefix}{item.title}\n"))
        fragments.append(("class:rail.meta", f"  {item.meta}"))
    return fragments


__all__ = [
    "SessionListItem",
    "empty_session_list_fragments",
    "format_updated_label",
    "item_from_summary",
    "item_from_session",
    "items_from_summaries",
    "render_session_list_fragments",
]
