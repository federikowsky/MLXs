from __future__ import annotations

from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.containers import FloatContainer
from prompt_toolkit.layout.menus import CompletionsMenu

from mlxs.chat.tui.scaffold import (
    BodyScaffoldParts,
    ShellScaffoldParts,
    build_body_scaffold,
    build_transcript_first_scaffold,
)


def test_build_body_scaffold_returns_center_when_no_rails() -> None:
    center = Window()

    body = build_body_scaffold(BodyScaffoldParts(center=center))

    assert body is center


def test_build_body_scaffold_returns_split_when_rails_present() -> None:
    left = Window()
    center = Window()
    right = Window()

    body = build_body_scaffold(BodyScaffoldParts(left=left, center=center, right=right))

    assert isinstance(body, VSplit)
    assert body.children[0] is left
    assert body.children[1] is center
    assert body.children[2] is right


def test_build_transcript_first_scaffold_preserves_current_structure() -> None:
    header = Window()
    body = Window()
    composer = Window()
    composer_meta = VSplit([Window()])
    footer = VSplit([Window()])
    menu = CompletionsMenu(max_height=10)

    scaffold = build_transcript_first_scaffold(
        ShellScaffoldParts(
            header=header,
            body=body,
            progress=None,
            composer=composer,
            composer_meta=composer_meta,
            footer=footer,
            completion_menu=menu,
        )
    )

    assert isinstance(scaffold, FloatContainer)
    assert isinstance(scaffold.content, HSplit)
    assert scaffold.content.children[0] is header
    assert scaffold.content.children[1] is body
    assert scaffold.content.children[3] is composer
    assert scaffold.content.children[4] is composer_meta
    assert scaffold.content.children[5] is footer
    assert len(scaffold.floats) == 1
    assert scaffold.floats[0].content is menu


def test_build_transcript_first_scaffold_accepts_progress_strip() -> None:
    header = Window()
    body = Window()
    progress = Window(height=1)
    composer = Window()
    composer_meta = VSplit([Window()])
    footer = VSplit([Window()])
    menu = CompletionsMenu(max_height=10)

    scaffold = build_transcript_first_scaffold(
        ShellScaffoldParts(
            header=header,
            body=body,
            progress=progress,
            composer=composer,
            composer_meta=composer_meta,
            footer=footer,
            completion_menu=menu,
        )
    )

    assert scaffold.content.children[2] is progress
