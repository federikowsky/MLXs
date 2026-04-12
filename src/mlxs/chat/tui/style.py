"""Prompt-toolkit style definitions for the chat TUI."""

from __future__ import annotations

from prompt_toolkit.styles import Style


CHAT_STYLE = Style.from_dict(
    {
        "shell": "bg:#0b0f14 #e6edf3",
        "surface": "bg:#0e141b",
        "header": "bg:#0e141b #e6edf3",
        "header.brand": "bg:#ff8a65 #0b0f14 bold",
        "header.model": "bg:#0e141b #f0f6fc bold",
        "header.meta": "bg:#0e141b #8b949e",
        "header.path": "bg:#0e141b #c9d1d9",
        "header.badge": "bg:#18212b #cdd9e5 bold",
        "rail": "bg:#0d1218 #b7c4d1",
        "rail.title": "bg:#0d1218 #e6edf3 bold",
        "rail.item": "bg:#0d1218 #c9d1d9",
        "rail.item.active": "bg:#101923 #f0f6fc bold",
        "rail.meta": "bg:#0d1218 #7d8590",
        "rail.filter": "bg:#0d1218 #c9d1d9",
        "rail.filter_prompt": "bg:#0d1218 #7d8590 bold",
        "transcript": "bg:#0b0f14 #e6edf3",
        "label.user": "#cbd5e1 bold",
        "label.assistant": "bg:#163225 #b8f7c0 bold",
        "label.system": "bg:#4f4017 #f5d06b bold",
        "label.status": "bg:#162a40 #9fd3ff bold",
        "label.error": "bg:#4a1f1f #ffb4ae bold",
        "label.help": "bg:#36224d #e2c8ff bold",
        "body.user": "#dbe4ee",
        "body.assistant": "#edf5ff",
        "body.system": "#f4d57a",
        "body.status": "#c9d1d9",
        "body.error": "#ffb4ae",
        "body.help": "#e2c8ff",
        "attachment": "#7d8590 italic",
        "composer": "bg:#111821 #f0f6fc",
        "composer.prompt": "#7dd3fc bold",
        "composer.context": "bg:#0f151c #a7b7c9",
        "composer.shortcuts": "bg:#0f151c #768390",
        "progress": "bg:#0f151c #c9d1d9",
        "progress.meta": "bg:#0f151c #a7b7c9",
        "progress.state": "bg:#0f151c #f0f6fc bold",
        "statusbar": "bg:#0f151c #c9d1d9",
        "statusbar.meta": "bg:#0f151c #8b949e",
        "statusbar.state": "bg:#0f151c #f0f6fc bold",
    }
)


__all__ = ["CHAT_STYLE"]
