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
class BodyScaffoldParts:
    """Named body-region parts for the future transcript/rail layout."""

    center: AnyContainer
    left: AnyContainer | None = None
    right: AnyContainer | None = None


@dataclass(frozen=True, slots=True)
class ShellScaffoldParts:
    """Named scaffold parts used to assemble the current chat shell layout."""

    header: AnyContainer
    body: AnyContainer
    progress: AnyContainer | None
    composer: AnyContainer
    composer_meta: VSplit
    footer: VSplit
    completion_menu: AnyContainer


def build_body_scaffold(parts: BodyScaffoldParts) -> AnyContainer:
    """Build the future main body region while preserving transcript dominance."""
    children: list[AnyContainer] = []
    if parts.left is not None:
        children.append(parts.left)
    children.append(parts.center)
    if parts.right is not None:
        children.append(parts.right)

    if len(children) == 1:
        return parts.center
    return VSplit(children)


def build_transcript_first_scaffold(parts: ShellScaffoldParts) -> FloatContainer:
    """Build the current transcript-first shell scaffold.

    This intentionally preserves the current vertical structure while making
    the scaffold explicit for later Phase 2 layout work.
    """
    body_children: list[AnyContainer] = [
        parts.header,
        parts.body,
    ]
    if parts.progress is not None:
        body_children.append(parts.progress)
    body_children.extend(
        [
            Window(height=1, char=" ", style="class:surface"),
            parts.composer,
            parts.composer_meta,
            parts.footer,
        ]
    )
    return FloatContainer(
        content=HSplit(body_children),
        floats=[
            Float(
                xcursor=True,
                ycursor=True,
                content=parts.completion_menu,
            ),
        ],
    )


__all__ = [
    "BodyScaffoldParts",
    "ShellScaffoldParts",
    "build_body_scaffold",
    "build_transcript_first_scaffold",
]
