"""Chat session data model — messages and session state.

Pure data: no I/O, no business logic, no UI dependency.
Serialisable to/from JSON for persistent storage.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class ChatMessage:
    """A single message in a chat session."""

    role: str  # "user", "assistant", "system"
    content: str
    created_at: str = field(default_factory=lambda: _now_iso())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(
            role=data["role"],
            content=data["content"],
            created_at=data.get("created_at", _now_iso()),
            metadata=data.get("metadata", {}),
        )

    def as_prompt_dict(self) -> dict[str, str]:
        """Return the minimal dict expected by chat template functions."""
        return {
            "role": self.role,
            "content": _prompt_content(self.role, self.content, self.metadata),
        }


@dataclass(slots=True)
class ChatSession:
    """A multi-turn chat session with metadata."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New chat"
    messages: list[ChatMessage] = field(default_factory=list)
    model_path: str = ""
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())
    generation_config: dict[str, Any] = field(default_factory=dict)

    # -- Mutation helpers --------------------------------------------------

    def add_user_message(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(role="user", content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._touch()
        return msg

    def add_assistant_message(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        self._touch()
        return msg

    def delete_last_turn(self) -> int:
        """Remove the last user+assistant exchange.  Returns number of messages removed."""
        removed = 0
        while self.messages and self.messages[-1].role == "assistant":
            self.messages.pop()
            removed += 1
        while self.messages and self.messages[-1].role == "user":
            self.messages.pop()
            removed += 1
        if removed:
            self._touch()
        return removed

    def clear_messages(self) -> None:
        self.messages.clear()
        self._touch()

    def set_system_message(self, content: str) -> None:
        """Insert or replace the leading system message."""
        text = content.strip()
        if self.messages and self.messages[0].role == "system":
            if text:
                self.messages[0].content = text
            else:
                self.messages.pop(0)
        elif text:
            self.messages.insert(0, ChatMessage(role="system", content=text))
        self._touch()

    def system_message(self) -> str | None:
        """Return the current system message, if present."""
        if self.messages and self.messages[0].role == "system":
            return self.messages[0].content
        return None

    def prompt_messages(self) -> list[dict[str, str]]:
        """Return messages in the format expected by template functions."""
        return [m.as_prompt_dict() for m in self.messages]

    # -- Auto-title --------------------------------------------------------

    def auto_title(self) -> None:
        """Set title from the first user message if still default."""
        if self.title != "New chat":
            return
        for msg in self.messages:
            if msg.role == "user" and msg.content.strip():
                text = msg.content.strip()
                self.title = text[:60] + ("..." if len(text) > 60 else "")
                return

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "model_path": self.model_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "generation_config": self.generation_config,
            "messages": [m.to_dict() for m in self.messages],
        }

    def to_markdown(self) -> str:
        """Render the session as a Markdown transcript."""
        lines = [
            f"# {self.title}",
            "",
            f"- Session: `{self.session_id}`",
            f"- Model: `{self.model_path or 'default'}`",
            f"- Created: `{self.created_at}`",
            f"- Updated: `{self.updated_at}`",
            "",
        ]
        for message in self.messages:
            heading = {
                "assistant": "Assistant",
                "system": "System",
                "user": "User",
            }.get(message.role, message.role.capitalize())
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(message.content.rstrip())
            attachments = _attachment_metadata(message.metadata)
            if attachments:
                lines.append("")
                lines.append("Attached files:")
                lines.extend(f"- `{attachment['path']}`" for attachment in attachments)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatSession:
        return cls(
            session_id=data.get("session_id", uuid.uuid4().hex[:12]),
            title=data.get("title", "New chat"),
            model_path=data.get("model_path", ""),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            generation_config=data.get("generation_config", {}),
            messages=[
                ChatMessage.from_dict(m) for m in data.get("messages", [])
            ],
        )

    # -- Internal ----------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = _now_iso()


def _now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _attachment_metadata(metadata: dict[str, Any]) -> list[dict[str, str]]:
    attachments = metadata.get("attachments")
    if not isinstance(attachments, list):
        return []
    return [
        attachment
        for attachment in attachments
        if isinstance(attachment, dict)
        and isinstance(attachment.get("path"), str)
        and isinstance(attachment.get("content"), str)
    ]


def _prompt_content(role: str, content: str, metadata: dict[str, Any]) -> str:
    if role != "user":
        return content
    attachments = _attachment_metadata(metadata)
    if not attachments:
        return content

    parts = [content.rstrip(), "", "Attached files:"]
    for attachment in attachments:
        path = attachment["path"]
        body = attachment["content"].rstrip()
        parts.extend(
            (
                f"[file] {path}",
                "----- BEGIN FILE -----",
                body,
                "----- END FILE -----",
                "",
            )
        )
    return "\n".join(parts).rstrip()
