"""Prompt-toolkit overlay host helpers for the chat TUI."""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.layout import Float, FloatContainer
from prompt_toolkit.layout.containers import AnyContainer


@dataclass(frozen=True, slots=True)
class OverlaySpec:
    """A generic overlay mounted in the shell's float host."""

    content: AnyContainer
    xcursor: bool = False
    ycursor: bool = False
    top: int | None = None
    right: int | None = None
    bottom: int | None = None
    left: int | None = None


def anchored_overlay(content: AnyContainer, *, xcursor: bool = False, ycursor: bool = False) -> OverlaySpec:
    """Build a simple cursor-anchored overlay spec."""
    return OverlaySpec(content=content, xcursor=xcursor, ycursor=ycursor)


def positioned_overlay(
    content: AnyContainer,
    *,
    top: int | None = None,
    right: int | None = None,
    bottom: int | None = None,
    left: int | None = None,
) -> OverlaySpec:
    """Build a fixed-position overlay spec."""
    return OverlaySpec(
        content=content,
        top=top,
        right=right,
        bottom=bottom,
        left=left,
    )


def build_overlay_host(content: AnyContainer, overlays: tuple[OverlaySpec, ...] = ()) -> FloatContainer:
    """Wrap shell content in a shared float host for future overlays."""
    return FloatContainer(
        content=content,
        floats=[
            Float(
                content=overlay.content,
                xcursor=overlay.xcursor,
                ycursor=overlay.ycursor,
                top=overlay.top,
                right=overlay.right,
                bottom=overlay.bottom,
                left=overlay.left,
            )
            for overlay in overlays
        ],
    )


__all__ = ["OverlaySpec", "anchored_overlay", "build_overlay_host", "positioned_overlay"]
