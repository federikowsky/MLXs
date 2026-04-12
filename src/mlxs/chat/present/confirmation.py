"""Pure presentation helpers for the confirmation dialog overlay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfirmationSpec:
    """Presentation spec for a single confirmation prompt."""

    title: str
    body: str
    confirm_label: str = "Confirm"
    cancel_label: str = "Esc cancel"


def render_confirmation_fragments(spec: ConfirmationSpec | None) -> list[tuple[str, str]]:
    """Render a compact confirmation dialog body."""
    if spec is None:
        return []

    fragments: list[tuple[str, str]] = [
        ("class:confirm.title", f" {spec.title} "),
        ("class:confirm.meta", f"  Enter {spec.confirm_label} · {spec.cancel_label}"),
    ]
    for line in spec.body.splitlines() or [""]:
        fragments.append(("", "\n"))
        fragments.append(("class:confirm.body", f"  {line}" if line else ""))
    return fragments


__all__ = ["ConfirmationSpec", "render_confirmation_fragments"]
