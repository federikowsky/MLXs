from __future__ import annotations

from mlxs.chat.present.sessions import (
    empty_session_list_fragments,
    format_updated_label,
    item_from_session,
    item_from_summary,
    items_from_summaries,
    render_session_list_fragments,
)
from mlxs.chat.session import ChatSession
from mlxs.chat.store import ChatSessionSummary


def test_item_from_summary_builds_title_meta_and_search_text() -> None:
    summary = ChatSessionSummary(
        session_id="abc123",
        title="Latency hypothesis",
        model_path="default",
        updated_at="2026-04-12T12:00:00+00:00",
        message_count=3,
        active=True,
    )

    item = item_from_summary(summary)

    assert item.session_id == "abc123"
    assert item.title == "Latency hypothesis"
    assert item.meta == "3 messages · 2026-04-12 12:00"
    assert item.active is True
    assert "latency hypothesis" in item.search_text


def test_items_from_summaries_filters_case_insensitively() -> None:
    summaries = [
        ChatSessionSummary(
            session_id="conv-a",
            title="Latency hypothesis",
            model_path="default",
            updated_at="2026-04-12T12:00:00+00:00",
            message_count=3,
            active=True,
        ),
        ChatSessionSummary(
            session_id="conv-b",
            title="Quantum basics",
            model_path="default",
            updated_at="2026-04-12T12:01:00+00:00",
            message_count=2,
            active=False,
        ),
    ]

    items = items_from_summaries(summaries, query="quant")

    assert len(items) == 1
    assert items[0].session_id == "conv-b"


def test_empty_session_list_fragments_render_empty_state_text() -> None:
    rendered = "".join(part for _, part in empty_session_list_fragments())

    assert "No conversations yet." in rendered
    assert "Start a chat to create the first session." in rendered


def test_item_from_session_uses_compact_updated_label() -> None:
    session = ChatSession(model_path="default")
    session.title = "Latency hypothesis"
    session.updated_at = "2026-04-12T12:00:00+00:00"

    item = item_from_session(session)

    assert item.title == "Latency hypothesis"
    assert "2026-04-12 12:00" in item.meta
    assert format_updated_label(session.updated_at) == "2026-04-12 12:00"


def test_render_session_list_fragments_marks_active_item() -> None:
    items = [
        item_from_summary(
            ChatSessionSummary(
                session_id="conv-a",
                title="Latency hypothesis",
                model_path="default",
                updated_at="2026-04-12T12:00:00+00:00",
                message_count=3,
                active=True,
            )
        ),
        item_from_summary(
            ChatSessionSummary(
                session_id="conv-b",
                title="Quantum basics",
                model_path="default",
                updated_at="2026-04-12T12:01:00+00:00",
                message_count=2,
                active=False,
            )
        ),
    ]

    rendered = "".join(text for _, text in render_session_list_fragments(items))

    assert "Conversations" in rendered
    assert "> Latency hypothesis" in rendered
    assert "Quantum basics" in rendered
