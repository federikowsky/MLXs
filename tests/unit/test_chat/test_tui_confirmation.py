from __future__ import annotations

from mlxs.chat.present.confirmation import ConfirmationSpec
from mlxs.chat.tui.confirmation import ConfirmationDialog


def test_confirmation_dialog_renders_confirmation_spec() -> None:
    dialog = ConfirmationDialog(invalidate=lambda: None)
    dialog.open(
        ConfirmationSpec(
            title="Clear conversation?",
            body="This removes the current transcript.",
            confirm_label="Clear",
        )
    )

    rendered = "".join(text for _, text in dialog.fragments())

    assert "Clear conversation?" in rendered
    assert "Enter Clear" in rendered
    assert "This removes the current transcript." in rendered
