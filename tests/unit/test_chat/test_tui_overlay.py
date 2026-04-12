from __future__ import annotations

from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.containers import FloatContainer

from mlxs.chat.tui.overlay import anchored_overlay, build_overlay_host


def test_build_overlay_host_wraps_content_without_overlays() -> None:
    content = HSplit([Window()])

    host = build_overlay_host(content)

    assert isinstance(host, FloatContainer)
    assert host.content is content
    assert host.floats == []


def test_build_overlay_host_mounts_cursor_anchored_overlays() -> None:
    content = HSplit([Window()])
    overlay_window = Window()

    host = build_overlay_host(
        content,
        overlays=(anchored_overlay(overlay_window, xcursor=True, ycursor=True),),
    )

    assert len(host.floats) == 1
    assert host.floats[0].content is overlay_window
    assert host.floats[0].xcursor is True
    assert host.floats[0].ycursor is True
