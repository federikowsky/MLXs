from __future__ import annotations

from mlxs.chat.present.references import ReferencePickerItem, render_reference_picker_fragments


def test_render_reference_picker_fragments_renders_selected_item() -> None:
    items = [
        ReferencePickerItem(label="src/main.py", insert_text="@src/main.py", meta="attach file"),
        ReferencePickerItem(label="docs/readme.md", insert_text="@docs/readme.md", meta="attach file"),
    ]

    rendered = "".join(
        text for _, text in render_reference_picker_fragments(items, selected_index=0, query="src")
    )

    assert "References" in rendered
    assert "@src/main.py" in rendered
    assert "attach file" in rendered


def test_render_reference_picker_fragments_handles_empty_state() -> None:
    rendered = "".join(text for _, text in render_reference_picker_fragments([], query="zzz"))

    assert "No matching files" in rendered
