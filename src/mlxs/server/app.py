"""ASGI application factory (§6.7).

Creates and configures the Starlette application with routes
and middleware.
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.routing import Route

from mlxs.server.deps import Dependencies
from mlxs.server.routes.chat import chat_completions
from mlxs.server.routes.health import health


def create_app(deps: Dependencies) -> Starlette:
    """Create the ASGI app with all routes wired.

    Args:
        deps: Pre-built dependency container from ``create_dependencies()``.

    Returns:
        Configured Starlette ASGI application.
    """
    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ]

    app = Starlette(routes=routes)
    app.state.deps = deps
    return app
