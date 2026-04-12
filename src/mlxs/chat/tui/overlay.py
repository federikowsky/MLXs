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


def anchored_overlay(content: AnyContainer, *, xcursor: bool = False, ycursor: bool = False) -> OverlaySpec:
    """Build a simple cursor-anchored overlay spec."""
    return OverlaySpec(content=content, xcursor=xcursor, ycursor=ycursor)


def build_overlay_host(content: AnyContainer, overlays: tuple[OverlaySpec, ...] = ()) -> FloatContainer:
    """Wrap shell content in a shared float host for future overlays."""
    return FloatContainer(
        content=content,
        floats=[
            Float(
                content=overlay.content,
                xcursor=overlay.xcursor,
                ycursor=overlay.ycursor,
            )
            for overlay in overlays
        ],
    )


__all__ = ["OverlaySpec", "anchored_overlay", "build_overlay_host"]
