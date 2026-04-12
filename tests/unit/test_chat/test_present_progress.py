from __future__ import annotations

from mlxs.chat.present.progress import build_progress_fragments, progress_visible


def test_progress_visible_only_for_non_idle_states() -> None:
    assert progress_visible(state="idle") is False
    assert progress_visible(state="generating") is True
    assert progress_visible(state="cancelling") is True
    assert progress_visible(state="error") is True


def test_build_progress_fragments_uses_real_shell_state_labels() -> None:
    rendered = "".join(text for _, text in build_progress_fragments(state="generating", detail="Streaming reply"))

    assert "Running" in rendered
    assert "Streaming reply" in rendered
