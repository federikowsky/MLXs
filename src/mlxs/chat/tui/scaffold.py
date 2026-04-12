"""Prompt-toolkit scaffold/layout assembly for the chat shell.

This module owns only the composition of prompt_toolkit containers. It does
not own business logic, controller flow, or visual formatting decisions beyond
the current transcript-first scaffold shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.layout import Float, FloatContainer, HSplit, VSplit, Window
from prompt_toolkit.layout.containers import AnyContainer


@dataclass(frozen=True, slots=True)
class ShellScaffoldParts:
    """Named scaffold parts used to assemble the current chat shell layout."""

    header: AnyContainer
    transcript: AnyContainer
    composer: AnyContainer
    composer_meta: VSplit
    footer: VSplit
    completion_menu: AnyContainer


def build_transcript_first_scaffold(parts: ShellScaffoldParts) -> FloatContainer:
    """Build the current transcript-first shell scaffold.

    This intentionally preserves the current vertical structure while making
    the scaffold explicit for later Phase 2 layout work.
    """
    return FloatContainer(
        content=HSplit(
            [
                parts.header,
                parts.transcript,
                Window(height=1, char=" ", style="class:surface"),
                parts.composer,
                parts.composer_meta,
                parts.footer,
            ]
        ),
        floats=[
            Float(
                xcursor=True,
                ycursor=True,
                content=parts.completion_menu,
            ),
        ],
    )


__all__ = ["ShellScaffoldParts", "build_transcript_first_scaffold"]
