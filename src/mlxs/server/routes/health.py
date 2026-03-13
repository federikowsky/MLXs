"""Health check endpoint (§6.7)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse


async def health(request: Request) -> JSONResponse:
    """GET /health — basic liveness check."""
    return JSONResponse({"status": "ok"})
