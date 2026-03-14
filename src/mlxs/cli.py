"""CLI entry point — dispatch to serve or chat.

Single entry point: mlxs serve | mlxs chat. Parsing and config resolution
in config/cli.py; business logic in server (app, chat) and ui/tui.
Errors are caught and only the message is shown (no stack trace to user).
"""

from __future__ import annotations

import sys

from mlxs._errors import MLXsError
from mlxs.config.cli import parse_argv
from mlxs.config.schema import AppConfig


def main() -> None:
    """Parse argv, resolve config, dispatch to serve or chat."""
    try:
        _main()
    except SystemExit:
        raise
    except MLXsError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _main() -> None:
    # Check for --simple-chat before parse_argv (it's not in the config schema)
    simple_chat = "--simple-chat" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--simple-chat"]

    try:
        subcommand, config = parse_argv(argv)
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 1)

    if subcommand in ("serve", "chat") and not (config.model.model_path or "").strip():
        print(
            "Model path is required. Use --model <path-or-hf-id>"
            " or set model.model_path in config.",
            file=sys.stderr,
        )
        sys.exit(1)

    if subcommand == "serve":
        _run_serve(config)
    elif subcommand == "chat":
        _run_chat(config, simple_chat=simple_chat)
    else:
        print(f"Unknown subcommand: {subcommand}", file=sys.stderr)
        sys.exit(1)


def _run_serve(config: AppConfig) -> None:
    """Start the HTTP server."""
    from mlxs.server.app import create_app
    from mlxs.server.deps import create_dependencies

    deps = create_dependencies(config)
    app = create_app(deps)
    host = config.server.host
    port = config.server.port
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def _run_chat(config: AppConfig, *, simple_chat: bool = False) -> None:
    """Start interactive chat — TUI by default, --simple-chat for old loop."""
    from mlxs.server.deps import create_dependencies

    deps = create_dependencies(config)

    if simple_chat:
        from mlxs.server.chat import run_chat_loop
        run_chat_loop(deps)
    else:
        from mlxs.ui.tui.app import run_tui
        run_tui(deps)


if __name__ == "__main__":
    main()

