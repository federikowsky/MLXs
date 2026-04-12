"""Pure presentation helpers for shell chrome formatting."""

from __future__ import annotations

from pathlib import Path


def build_header_fragments(
    *,
    model_id: str,
    title: str,
    cwd_label: str,
    branch: str | None,
    turns: int,
    system_on: bool,
) -> list[tuple[str, str]]:
    model = short_model_name(model_id)
    branch_label = branch or "-"
    system_label = "system on" if system_on else "system off"
    return [
        ("class:header.brand", " MLXs "),
        ("class:header.model", f"  {model}  "),
        ("class:header.path", f"  {cwd_label}\n"),
        ("class:header.badge", f" {branch_label} "),
        ("class:header.meta", f"  {title}  "),
        ("class:header.meta", f"{turns} turns  "),
        ("class:header.badge", f" {system_label} "),
    ]


def build_composer_context_fragments(
    *,
    text: str,
    state: str,
    title: str,
    system_on: bool,
    palette_open: bool = False,
    reference_picker_open: bool = False,
    command_context: str | None = None,
    mention_paths: list[str] | None = None,
) -> list[tuple[str, str]]:
    if state == "generating":
        return [("class:composer.context", " assistant is responding live")]
    if state == "cancelling":
        return [("class:composer.context", " stopping the current turn cleanly")]
    if reference_picker_open:
        return [("class:composer.context", " reference picker open · filter files and insert one into the composer")]
    if palette_open:
        return [("class:composer.context", " command palette open · filter commands and insert one into the composer")]

    stripped = text.strip()
    if not stripped:
        context = (
            f" working in {title} · system {'on' if system_on else 'off'}"
            " · ask anything or insert files with @"
        )
        return [("class:composer.context", context)]

    if stripped == "?":
        return [
            (
                "class:composer.context",
                " help shortcut ready · press Enter to open commands and keys",
            )
        ]

    if stripped.startswith("/") and command_context is not None:
        return [("class:composer.context", f" {command_context}")]

    mentions = mention_paths or []
    if mentions:
        preview = ", ".join(mentions[:3])
        extra = f" +{len(mentions) - 3}" if len(mentions) > 3 else ""
        return [
            (
                "class:composer.context",
                f" {len(mentions)} file reference(s) ready · {preview}{extra}",
            )
        ]

    return [("class:composer.context", f" drafting a request in {title}")]


def build_composer_shortcuts_fragments(
    *,
    state: str,
    palette_open: bool = False,
    reference_picker_open: bool = False,
) -> list[tuple[str, str]]:
    if state == "generating":
        text = "Esc cancel · Ctrl+C stop"
    elif state == "cancelling":
        text = "Waiting for generation to stop..."
    elif reference_picker_open:
        text = "Type to filter · Up/Down move · Enter insert · Esc close"
    elif palette_open:
        text = "Type to filter · Up/Down move · Enter insert · Esc close"
    else:
        text = "/ palette · @ picker · Enter newline · Ctrl+J send · Ctrl+↑/↓ switch"
    return [("class:composer.shortcuts", f" {text}")]


def build_footer_left_fragments(
    *,
    model_id: str,
    branch: str | None,
    turns: int,
    system_on: bool,
    max_tokens: int,
    temperature: float,
) -> list[tuple[str, str]]:
    left = (
        f" {short_model_name(model_id)}"
        f" · {branch or '-'}"
        f" · {turns} turns"
        f" · system {'on' if system_on else 'off'}"
        f" · max {max_tokens}"
        f" · temp {temperature}"
    )
    return [("class:statusbar", left)]


def build_footer_right_fragments(*, state: str, detail: str) -> list[tuple[str, str]]:
    state_label = {
        "idle": "Ready",
        "generating": "Streaming",
        "cancelling": "Stopping",
        "error": "Attention",
    }.get(state, state.replace("_", " ").title())
    return [
        ("class:statusbar.meta", detail + "  "),
        ("class:statusbar.state", state_label),
    ]


def display_path(path: Path) -> str:
    home = Path.home()
    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    if not relative.parts:
        return "~"
    return f"~/{relative}"


def short_model_name(model_id: str) -> str:
    if not model_id:
        return "default"
    return model_id.split("/")[-1]


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


__all__ = [
    "build_composer_context_fragments",
    "build_composer_shortcuts_fragments",
    "build_footer_left_fragments",
    "build_footer_right_fragments",
    "build_header_fragments",
    "display_path",
    "short_model_name",
    "truncate_text",
]
