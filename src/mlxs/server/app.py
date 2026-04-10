"""Compatibility ASGI app factory delegating to the Layer 4 HTTP surface."""

from mlxs.product_surfaces.http import create_app

__all__ = ["create_app"]
