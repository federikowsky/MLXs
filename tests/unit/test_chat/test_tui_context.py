from __future__ import annotations

from mlxs.chat.present.context import build_context_rail_summary
from mlxs.chat.session import ChatSession
from mlxs.chat.tui.context import ContextRail


def test_context_rail_renders_summary_fragments() -> None:
    rail = ContextRail()
    session = ChatSession(title="Latency hypothesis", model_path="z-lab/Qwen3.5-2B-PARO")
    session.add_user_message("Check current session context")
    session.updated_at = "2026-04-12T12:34:00+00:00"
    rail.set_summary(
        build_context_rail_summary(
            session=session,
            model_id="z-lab/Qwen3.5-2B-PARO",
            state="generating",
            detail="Streaming reply",
        )
    )

    rendered = "".join(text for _, text in rail.fragments())

    assert "Context" in rendered
    assert "Latency hypothesis" in rendered
    assert "updated" in rendered
    assert "turns" in rendered
    assert "Running" in rendered
