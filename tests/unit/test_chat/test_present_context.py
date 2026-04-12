from __future__ import annotations

from mlxs.chat.present.context import build_context_rail_summary, render_context_rail_fragments
from mlxs.chat.session import ChatSession


def test_build_context_rail_summary_uses_real_session_and_state() -> None:
    session = ChatSession(title="Latency investigation", model_path="z-lab/Qwen3.5-2B-PARO")
    session.add_user_message("Why is the scheduler slower?")

    summary = build_context_rail_summary(
        session=session,
        model_id="z-lab/Qwen3.5-2B-PARO",
        state="generating",
        detail="Streaming reply",
    )

    labels = [row.label for row in summary.rows]
    values = [row.value for row in summary.rows]

    assert labels[:3] == ["chat", "session", "messages"]
    assert "Latency investigation" in values
    assert "1" in values
    assert "Running" in values
    assert "Streaming reply" in values


def test_render_context_rail_fragments_renders_empty_state() -> None:
    rendered = "".join(text for _, text in render_context_rail_fragments(None))

    assert "Context" in rendered
    assert "no active session" in rendered
