from __future__ import annotations

from mlxs.chat.session import ChatSession
from mlxs.chat.store import ChatSessionStore


def test_create_session_sets_active_session() -> None:
    store = ChatSessionStore()

    session = store.create_session(model_path="default")

    assert store.active_session_id == session.session_id
    assert store.get_active_session() is session


def test_switch_active_session_changes_selection() -> None:
    store = ChatSessionStore()
    first = store.create_session(model_path="a")
    second = store.create_session(model_path="b")

    active = store.switch_session(first.session_id)

    assert active is first
    assert store.active_session_id == first.session_id
    assert store.get_active_session() is first
    assert second.session_id != store.active_session_id


def test_list_summaries_reports_active_and_message_counts() -> None:
    store = ChatSessionStore()
    first = store.create_session(model_path="a")
    first.add_user_message("hello")
    first.add_assistant_message("world")
    store.update_session(first, make_active=True)

    second = store.create_session(model_path="b")
    summaries = store.list_summaries()

    assert len(summaries) == 2
    assert any(summary.active and summary.session_id == second.session_id for summary in summaries)
    assert any(
        (not summary.active) and summary.session_id == first.session_id and summary.message_count == 2
        for summary in summaries
    )


def test_update_session_reflects_title_and_updated_metadata() -> None:
    store = ChatSessionStore()
    session = store.create_session(model_path="default")
    session.title = "Renamed"
    session.updated_at = "2099-01-01T00:00:00+00:00"

    store.update_session(session, make_active=True)

    summary = store.list_summaries()[0]
    assert summary.title == "Renamed"
    assert summary.updated_at == "2099-01-01T00:00:00+00:00"


def test_update_session_can_register_existing_session() -> None:
    store = ChatSessionStore()
    session = ChatSession(model_path="default")

    store.update_session(session, make_active=True)

    assert store.get_active_session() is session
