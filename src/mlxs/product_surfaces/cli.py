"""Layer 4 CLI surface."""

from __future__ import annotations

import errno
import socket
import sys

from mlxs._errors import MLXsError
from mlxs.config.schema import AppConfig
from mlxs.product_surfaces.bootstrap import create_runtime
from mlxs.product_surfaces.config import parse_product_argv
from mlxs.product_surfaces.http import create_app


def main(argv: list[str] | None = None) -> None:
    """Parse argv, resolve config, and dispatch the Layer 4 product CLI."""
    try:
        _main(argv)
    except SystemExit:
        raise
    except MLXsError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _main(argv: list[str] | None = None) -> None:
    try:
        subcommand, config, chat_query = parse_product_argv(argv)
    except SystemExit as exc:
        sys.exit(exc.code if exc.code is not None else 1)

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
        _run_chat(config, initial_query=chat_query)
    else:
        print(f"Unknown subcommand: {subcommand}", file=sys.stderr)
        sys.exit(1)


def _run_serve(config: AppConfig) -> None:
    """Start the Layer 4 HTTP surface."""
    _assert_server_endpoint_available(config.server.host, config.server.port)
    runtime = create_runtime(config)
    app = create_app(runtime)
    import uvicorn

    uvicorn.run(app, host=config.server.host, port=config.server.port)


def _run_chat(config: AppConfig, *, initial_query: str | None = None) -> None:
    """Start the Layer 4 CLI chat surface."""
    runtime = create_runtime(config)
    from mlxs.product_surfaces.chat import run_chat_loop

    run_chat_loop(runtime, initial_query=initial_query)


def _assert_server_endpoint_available(host: str, port: int) -> None:
    """Fail fast on address-in-use before paying model bootstrap cost."""
    try:
        addrinfos = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
    except OSError as exc:
        raise MLXsError(f"Cannot resolve server address {host}:{port}: {exc}") from exc

    saw_in_use = False
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in addrinfos:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            return
        except OSError as exc:
            last_error = exc
            if exc.errno == errno.EADDRINUSE:
                saw_in_use = True
        finally:
            sock.close()

    if saw_in_use:
        raise MLXsError(f"Cannot start server: address {host}:{port} is already in use.")
    if last_error is not None:
        raise MLXsError(f"Cannot bind server address {host}:{port}: {last_error}") from last_error
