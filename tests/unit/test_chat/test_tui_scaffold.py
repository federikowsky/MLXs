from __future__ import annotations

from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.containers import FloatContainer
from prompt_toolkit.layout.menus import CompletionsMenu

from mlxs.chat.tui.scaffold import ShellScaffoldParts, build_transcript_first_scaffold


def test_build_transcript_first_scaffold_preserves_current_structure() -> None:
    header = Window()
    transcript = Window()
    composer = Window()
    composer_meta = VSplit([Window()])
    footer = VSplit([Window()])
    menu = CompletionsMenu(max_height=10)

    scaffold = build_transcript_first_scaffold(
        ShellScaffoldParts(
            header=header,
            transcript=transcript,
            composer=composer,
            composer_meta=composer_meta,
            footer=footer,
            completion_menu=menu,
        )
    )

    assert isinstance(scaffold, FloatContainer)
    assert isinstance(scaffold.content, HSplit)
    assert scaffold.content.children[0] is header
    assert scaffold.content.children[1] is transcript
    assert scaffold.content.children[3] is composer
    assert scaffold.content.children[4] is composer_meta
    assert scaffold.content.children[5] is footer
    assert len(scaffold.floats) == 1
    assert scaffold.floats[0].content is menu
