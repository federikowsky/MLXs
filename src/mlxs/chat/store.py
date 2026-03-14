"""Persistent session store — JSON files in ~/.mlxs/chat_sessions/.

Supports: list, load, save, rename, delete, create.
Each session is stored as a single JSON file named by session_id.
No external DB, no locking complexity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mlxs.chat.session import ChatSession

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".mlxs" / "chat_sessions"


class SessionStore:
    """File-backed session store."""

    __slots__ = ("_dir",)

    def __init__(self, directory: Path | str | None = None) -> None:
        self._dir = Path(directory) if directory else _DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- CRUD --------------------------------------------------------------

    def create(self, model_path: str = "") -> ChatSession:
        """Create a new empty session and persist it."""
        session = ChatSession(model_path=model_path)
        self.save(session)
        return session

    def save(self, session: ChatSession) -> Path:
        """Persist session to disk. Returns the file path."""
        path = self._path_for(session.session_id)
        data = session.to_dict()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        logger.debug("Saved session %s to %s", session.session_id, path)
        return path

    def load(self, session_id: str) -> ChatSession:
        """Load a session by id. Raises FileNotFoundError if missing."""
        path = self._path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChatSession.from_dict(data)

    def delete(self, session_id: str) -> bool:
        """Delete a session file. Returns True if deleted, False if not found."""
        path = self._path_for(session_id)
        if path.exists():
            path.unlink()
            logger.debug("Deleted session %s", session_id)
            return True
        return False

    def rename(self, session_id: str, new_title: str) -> ChatSession:
        """Rename a session's title and persist."""
        session = self.load(session_id)
        session.title = new_title
        self.save(session)
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions (summary only: id, title, model, dates, msg count).

        Returns list sorted by updated_at descending (most recent first).
        """
        summaries: list[dict[str, Any]] = []
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append({
                    "session_id": data.get("session_id", path.stem),
                    "title": data.get("title", "Untitled"),
                    "model_path": data.get("model_path", ""),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                })
            except Exception as exc:
                logger.warning("Skipping corrupt session file %s: %s", path, exc)
        summaries.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return summaries

    # -- Internal ----------------------------------------------------------

    def _path_for(self, session_id: str) -> Path:
        # Sanitise id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self._dir / f"{safe_id}.json"
