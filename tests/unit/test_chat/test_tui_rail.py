from __future__ import annotations

from prompt_toolkit.layout import Window

from mlxs.chat.present.sessions import item_from_summary
from mlxs.chat.store import ChatSessionSummary
from mlxs.chat.tui.rail import SessionRail, build_session_rail_window, session_rail_fragments


def test_session_rail_fragments_render_active_item() -> None:
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
        )
    ]

    rendered = "".join(text for _, text in session_rail_fragments(items))

    assert "Conversations" in rendered
    assert "Latency hypothesis" in rendered


def test_build_session_rail_window_returns_window() -> None:
    window = build_session_rail_window(lambda: [])
    assert isinstance(window, Window)


def test_session_rail_owns_items_and_fragments() -> None:
    rail = SessionRail()
    rail.set_items(
        [
            item_from_summary(
                ChatSessionSummary(
                    session_id="conv-a",
                    title="Latency hypothesis",
                    model_path="default",
                    updated_at="2026-04-12T12:00:00+00:00",
                    message_count=3,
                    active=True,
                )
            )
        ]
    )

    rendered = "".join(text for _, text in rail.fragments())

    assert "Conversations" in rendered
    assert "Latency hypothesis" in rendered


def test_session_rail_filters_items_from_filter_buffer() -> None:
    rail = SessionRail()
    rail.set_items(
        [
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
    )
    rail.filter_buffer.text = "quant"

    rendered = "".join(text for _, text in rail.fragments())

    assert "Quantum basics" in rendered
    assert "Latency hypothesis" not in rendered
