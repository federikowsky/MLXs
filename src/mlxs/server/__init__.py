"""Server module — HTTP API with SSE streaming (§6.7, §9).

Public API: ``create_app()`` — builds the ASGI application.
Composition root: ``create_dependencies()`` — wires all concrete
implementations.
"""

from mlxs.server.app import create_app
from mlxs.server.deps import Dependencies, create_dependencies

__all__ = ["Dependencies", "create_app", "create_dependencies"]
