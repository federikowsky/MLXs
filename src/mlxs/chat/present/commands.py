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


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("help",    "/help",                   "Show commands and keyboard shortcuts."),
    CommandSpec("model",   "/model",                  "Show the active model and runtime settings."),
    CommandSpec("new",     "/new",                    "Start a fresh chat session."),
    CommandSpec("clear",   "/clear",                  "Clear the current conversation."),
    CommandSpec("undo",    "/undo",                   "Remove the last user/assistant turn."),
    CommandSpec("retry",   "/retry",                  "Regenerate the last user turn."),
    CommandSpec("system",  "/system <text>|clear",    "Set, inspect, or clear the system prompt."),
    CommandSpec("title",   "/title <text>",           "Rename the current chat session."),
    CommandSpec("history", "/history [n]",            "Show recent transcript lines."),
    CommandSpec("export",  "/export [path]",          "Write the current transcript to Markdown."),
    CommandSpec("status",  "/status",                 "Alias for /stats."),
    CommandSpec("stats",   "/stats",                  "Show the current session summary."),
    CommandSpec("quit",    "/quit",                   "Exit the chat."),
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
        return "command mode · Tab shows the full command menu"
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
