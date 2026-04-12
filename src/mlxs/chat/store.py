"""Minimal in-memory conversation/session store for chat product surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from mlxs.chat.session import ChatSession


@dataclass(frozen=True, slots=True)
class ChatSessionSummary:
    """Lightweight summary for session list/state ownership."""

    session_id: str
    title: str
    model_path: str
    updated_at: str
    message_count: int
    active: bool


class ChatSessionStore:
    """In-memory owner of chat sessions and active-session selection."""

    __slots__ = ("_active_session_id", "_sessions")

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._active_session_id: str | None = None

    @property
    def active_session_id(self) -> str | None:
        return self._active_session_id

    def create_session(self, *, model_path: str = "") -> ChatSession:
        session = ChatSession(model_path=model_path)
        self._sessions[session.session_id] = session
        self._active_session_id = session.session_id
        return session

    def update_session(self, session: ChatSession, *, make_active: bool = False) -> None:
        self._sessions[session.session_id] = session
        if self._active_session_id is None or make_active:
            self._active_session_id = session.session_id

    def get_active_session(self) -> ChatSession:
        if self._active_session_id is None:
            raise LookupError("No active chat session.")
        return self._sessions[self._active_session_id]

    def switch_session(self, session_id: str) -> ChatSession:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        self._active_session_id = session_id
        return self._sessions[session_id]

    def list_summaries(self) -> list[ChatSessionSummary]:
        active = self._active_session_id
        sessions = sorted(
            self._sessions.values(),
            key=lambda session: (session.session_id == active, session.updated_at),
            reverse=True,
        )
        return [
            ChatSessionSummary(
                session_id=session.session_id,
                title=session.title,
                model_path=session.model_path,
                updated_at=session.updated_at,
                message_count=len(session.messages),
                active=session.session_id == active,
            )
            for session in sessions
        ]
