from __future__ import annotations

from pathlib import Path

from mlxs.chat.tui.reference_picker import ReferencePicker


def test_reference_picker_filters_file_candidates(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "src" / "misc.txt").write_text("hello\n", encoding="utf-8")

    picker = ReferencePicker(cwd=lambda: tmp_path, invalidate=lambda: None)
    picker.open(replace_start=0, replace_end=0, initial_query="src/m")

    labels = [item.label for item in picker.items()]

    assert labels == ["src/main.py", "src/misc.txt"]


def test_reference_picker_selected_item_returns_insert_text(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    picker = ReferencePicker(cwd=lambda: tmp_path, invalidate=lambda: None)
    picker.open(replace_start=3, replace_end=3, initial_query="src/m")

    selected = picker.selected_item()

    assert selected is not None
    assert selected.insert_text == "@src/main.py"
