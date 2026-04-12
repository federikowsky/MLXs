"""Command metadata, help card text, and live composer command-context.

Single source of truth for:
  - ``CommandSpec`` — declarative description of one slash command.
  - ``COMMANDS`` — the ordered tuple of all registered slash commands.
  - ``help_card()`` — full formatted help text for display.
  - ``command_context(text)`` — live context hint string keyed to the
    current composer text (used in the composer context row).
  - ``command_names()`` — slash-command name list for completion wiring.

No prompt_toolkit imports.  No I/O.  No business logic.

Authority: presentation layer — pure text formatting only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    """Declarative description for a slash command."""

    name: str
    usage: str
    description: str
    category: str


@dataclass(frozen=True)
class CommandPaletteItem:
    """Presentation item for one command in the palette."""

    name: str
    usage: str
    description: str
    category: str
    insert_text: str
    search_text: str


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("help",    "/help",                   "Show commands and keyboard shortcuts.", "General"),
    CommandSpec("quit",    "/quit",                   "Exit the chat.", "General"),
    CommandSpec("new",     "/new",                    "Start a fresh chat session.", "Conversation"),
    CommandSpec("clear",   "/clear",                  "Clear the current conversation.", "Conversation"),
    CommandSpec("undo",    "/undo",                   "Remove the last user/assistant turn.", "Conversation"),
    CommandSpec("retry",   "/retry",                  "Regenerate the last user turn.", "Conversation"),
    CommandSpec("title",   "/title <text>",           "Rename the current chat session.", "Conversation"),
    CommandSpec("system",  "/system <text>|clear",    "Set, inspect, or clear the system prompt.", "Session"),
    CommandSpec("history", "/history [n]",            "Show recent transcript lines.", "Session"),
    CommandSpec("export",  "/export [path]",          "Write the current transcript to Markdown.", "Session"),
    CommandSpec("status",  "/status",                 "Alias for /stats.", "Session"),
    CommandSpec("stats",   "/stats",                  "Show the current session summary.", "Session"),
    CommandSpec("model",   "/model",                  "Show the active model and runtime settings.", "Runtime"),
)


def help_card() -> str:
    """Return the full help card text (commands + keyboard shortcuts)."""
    width = max(len(spec.usage) for spec in COMMANDS)
    lines = ["slash commands:"]
    lines.extend(f"  {spec.usage.ljust(width)}  {spec.description}" for spec in COMMANDS)
    lines.extend(
        (
            "",
            "keyboard:",
            "  /              open command palette when composer is empty",
            "  @              open reference picker at a mention boundary",
            "  Enter          insert newline",
            "  Ctrl+J         submit current input",
            "  Ctrl+Up/Down   switch active conversation",
            "  Tab / Shift-Tab browse completion menu",
            "  Up / Down      recall previous inputs when composer is empty",
            "  Esc            cancel current generation",
            "  Ctrl+L         clear and redraw terminal",
            "  Ctrl+C         exit when idle, cancel when generating",
        )
    )
    return "\n".join(lines)


def command_context(text: str) -> str:
    """Return the live context hint string for *text* as typed in the composer.

    Called on every keystroke when the composer text starts with ``/``.
    """
    if text == "/":
        return "command palette trigger · press / from an empty composer to browse commands"
    matches = [spec for spec in COMMANDS if f"/{spec.name}".startswith(text)]
    if not matches:
        return "unknown command · press Tab to inspect available commands"
    if len(matches) == 1 and f"/{matches[0].name}" == text:
        spec = matches[0]
        return f"{spec.usage} · {spec.description}"
    preview = " · ".join(f"{spec.usage} {spec.description}" for spec in matches[:3])
    if len(matches) > 3:
        preview += " · ..."
    return preview


def command_names() -> tuple[str, ...]:
    """Return slash-command names in the format expected by completers."""
    return tuple(f"/{spec.name}" for spec in COMMANDS)


def command_palette_items(query: str = "") -> list[CommandPaletteItem]:
    """Return palette items filtered by the current query."""
    text = query.strip().lower()
    items = [
        CommandPaletteItem(
            name=spec.name,
            usage=spec.usage,
            description=spec.description,
            category=spec.category,
            insert_text=_command_insert_text(spec),
            search_text=f"{spec.name} {spec.usage} {spec.description} {spec.category}".lower(),
        )
        for spec in COMMANDS
    ]
    if not text:
        return items
    return [item for item in items if text in item.search_text]


def render_command_palette_fragments(
    items: list[CommandPaletteItem],
    *,
    selected_index: int = 0,
    query: str = "",
) -> list[tuple[str, str]]:
    """Render a calm, compact command palette list."""
    fragments: list[tuple[str, str]] = [
        ("class:palette.title", " Commands "),
        ("class:palette.meta", "  Type to filter · Enter to insert · Esc to close"),
    ]
    if not items:
        fragments.append(("", "\n"))
        message = "  No matching commands" if query.strip() else "  No commands available"
        fragments.append(("class:palette.empty", message))
        return fragments

    index = min(max(selected_index, 0), len(items) - 1)
    current_category: str | None = None
    for position, item in enumerate(items):
        if item.category != current_category:
            current_category = item.category
            fragments.append(("", "\n"))
            fragments.append(("class:palette.section", f" {current_category}"))
        title_style = "class:palette.item.active" if position == index else "class:palette.item"
        meta_style = "class:palette.meta.active" if position == index else "class:palette.meta"
        prefix = "> " if position == index else "  "
        fragments.append(("", "\n"))
        fragments.append((title_style, f"{prefix}{item.usage}\n"))
        fragments.append((meta_style, f"  {item.description}"))
    return fragments


def _command_insert_text(spec: CommandSpec) -> str:
    if " " in spec.usage:
        return f"/{spec.name} "
    return f"/{spec.name}"
