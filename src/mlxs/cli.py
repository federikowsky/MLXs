"""CLI entry point — dispatch to serve or chat (plan-chat-cli).

Single entry point: mlxs serve | mlxs chat. Parsing and config resolution
in config/cli.py; business logic in server (app, chat).
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
    except SystemExit as e:
        raise
    except MLXsError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _main() -> None:
    try:
        subcommand, config = parse_argv()
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 1)

    if subcommand in ("serve", "chat") and not (config.model.model_path or "").strip():
        print("Model path is required. Use --model <path-or-hf-id> or set model.model_path in config.", file=sys.stderr)
        sys.exit(1)

    if subcommand == "serve":
        _run_serve(config)
    elif subcommand == "chat":
        _run_chat(config)
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


def _run_chat(config: AppConfig) -> None:
    """Start interactive chat loop."""
    from mlxs.server.chat import run_chat_loop
    from mlxs.server.deps import create_dependencies

    deps = create_dependencies(config)
    run_chat_loop(deps)


if __name__ == "__main__":
    main()
