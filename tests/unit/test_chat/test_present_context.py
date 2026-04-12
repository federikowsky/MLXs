from __future__ import annotations

from mlxs.chat.present.context import build_context_rail_summary, render_context_rail_fragments
from mlxs.chat.session import ChatSession


def test_build_context_rail_summary_uses_real_session_and_state() -> None:
    session = ChatSession(title="Latency investigation", model_path="z-lab/Qwen3.5-2B-PARO")
    session.add_user_message(
        "Why is the scheduler slower?",
        metadata={
            "attachments": [
                {
                    "path": "src/mlxs/batch/scheduler.py",
                    "absolute_path": "/tmp/src/mlxs/batch/scheduler.py",
                    "content": "print('hi')",
                }
            ]
        },
    )
    session.add_assistant_message("Investigating current batching path.")
    session.updated_at = "2026-04-12T12:34:00+00:00"

    summary = build_context_rail_summary(
        session=session,
        model_id="z-lab/Qwen3.5-2B-PARO",
        state="generating",
        detail="Streaming reply",
    )

    labels = [row.label for row in summary.rows]
    values = {row.label: row.value for row in summary.rows}

    assert labels[:4] == ["chat", "session", "updated", "turns"]
    assert values["chat"] == "Latency investigation"
    assert values["updated"] == "2026-04-12 12:34"
    assert values["turns"] == "1 user · 1 asst"
    assert values["files"] == "1"
    assert values["state"] == "Running"
    assert values["detail"] == "Streaming reply"


def test_render_context_rail_fragments_renders_empty_state() -> None:
    rendered = "".join(text for _, text in render_context_rail_fragments(None))

    assert "Context" in rendered
    assert "no active session" in rendered
