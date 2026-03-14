"""Tests for chat session metadata and prompt rendering."""

from __future__ import annotations

from mlxs.chat.session import ChatSession


def test_prompt_messages_include_attached_files_without_mutating_raw_text() -> None:
    session = ChatSession(model_path="default")
    session.add_user_message(
        "Review this file",
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

    prompt_messages = session.prompt_messages()

    assert session.messages[0].content == "Review this file"
    assert "Attached files:" in prompt_messages[0]["content"]
    assert "[file] src/app.py" in prompt_messages[0]["content"]
    assert "print('hi')" in prompt_messages[0]["content"]


def test_markdown_export_lists_attached_files() -> None:
    session = ChatSession(model_path="default")
    session.add_user_message(
        "Review this file",
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
    session.add_assistant_message("Looks good.")

    exported = session.to_markdown()

    assert "Attached files:" in exported
    assert "- `src/app.py`" in exported
