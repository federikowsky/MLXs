from __future__ import annotations

from mlxs.chat.present.transcript import TranscriptEntry, empty_state_fragments, entry_from_message, render_entry
from mlxs.chat.session import ChatMessage


def test_entry_from_message_carries_attachment_paths() -> None:
    message = ChatMessage(
        role="user",
        content="Review this file",
        metadata={
            "attachments": [
                {
                    "path": "src/app.py",
                    "absolute_path": "/tmp/src/app.py",
                    "content": "print('hi')\n",
                }
            ]
        },
    )

    entry = entry_from_message(message)

    assert entry.kind == "user"
    assert entry.attachments == ("src/app.py",)


def test_render_entry_for_pending_assistant_shows_live_badge() -> None:
    entry = TranscriptEntry(kind="assistant", title="assistant", body="Hello world")

    rendered = "".join(part for _, part in render_entry(entry, pending=True))

    assert "live" in rendered
    assert "Hello world" in rendered


def test_empty_state_fragments_include_examples() -> None:
    rendered = "".join(part for _, part in empty_state_fragments())

    assert "What can I help you ship today?" in rendered
    assert "/help" in rendered
