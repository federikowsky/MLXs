"""Layer 4 HTTP / SSE / metrics surfaces."""

from __future__ import annotations

from contextlib import asynccontextmanager
import inspect
from typing import Any

from mlxs.product_surfaces.observability import metrics_endpoint


def _surface_runtime(request: Any) -> Any:
    return request.app.state.runtime


def _metrics_enabled(runtime: Any) -> bool:
    config = getattr(runtime, "config", None)
    observability = getattr(config, "observability", None)
    return bool(observability and observability.metrics_enabled)


async def health(request: Any) -> Any:
    """Layer 4 health/readiness surface."""
    from starlette.responses import JSONResponse

    runtime = _surface_runtime(request)
    lifecycle = getattr(runtime, "lifecycle", None)
    if lifecycle is None:
        return JSONResponse({"status": "ok"})
    return JSONResponse(
        lifecycle.health_payload(
            metrics_enabled=_metrics_enabled(runtime),
            metrics_route_enabled=_metrics_enabled(runtime),
        )
    )


def create_app(runtime: Any) -> Any:
    """Create the Layer 4 HTTP app from the prebuilt product runtime."""
    from starlette.applications import Starlette
    from starlette.routing import Route

    from mlxs.server.routes.chat import chat_completions

    @asynccontextmanager
    async def lifespan(app: Any):
        try:
            yield
        finally:
            shutdown = getattr(runtime, "shutdown", None)
            if callable(shutdown):
                result = shutdown()
                if inspect.isawaitable(result):
                    await result

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ]
    if _metrics_enabled(runtime):
        routes.append(Route("/metrics", metrics_endpoint, methods=["GET"]))

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.runtime = runtime
    return app
