"""Pure presentation helpers for the compact @ reference picker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferencePickerItem:
    """Presentation item for one reference-picker result."""

    label: str
    insert_text: str
    meta: str


def render_reference_picker_fragments(
    items: list[ReferencePickerItem],
    *,
    selected_index: int = 0,
    query: str = "",
) -> list[tuple[str, str]]:
    """Render a calm, compact @ picker list."""
    fragments: list[tuple[str, str]] = [
        ("class:picker.title", " References "),
        ("class:picker.meta", "  Type to filter files · Enter to insert · Esc to close"),
    ]
    if not items:
        fragments.append(("", "\n"))
        message = "  No matching files" if query.strip() else "  No files available"
        fragments.append(("class:picker.empty", message))
        return fragments

    index = min(max(selected_index, 0), len(items) - 1)
    for position, item in enumerate(items):
        title_style = "class:picker.item.active" if position == index else "class:picker.item"
        meta_style = "class:picker.meta.active" if position == index else "class:picker.meta"
        prefix = "> " if position == index else "  "
        fragments.append(("", "\n"))
        fragments.append((title_style, f"{prefix}@{item.label}\n"))
        fragments.append((meta_style, f"  {item.meta}"))
    return fragments


__all__ = ["ReferencePickerItem", "render_reference_picker_fragments"]
