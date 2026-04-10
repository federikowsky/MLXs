"""Layer 4 / Product Surfaces canonical boundary."""

from __future__ import annotations

__all__ = [
    "ProductRuntime",
    "create_app",
    "create_runtime",
    "main",
]


def __getattr__(name: str) -> object:
    if name in {"ProductRuntime", "create_runtime"}:
        from mlxs.product_surfaces.bootstrap import ProductRuntime, create_runtime

        return {
            "ProductRuntime": ProductRuntime,
            "create_runtime": create_runtime,
        }[name]
    if name == "create_app":
        from mlxs.product_surfaces.http import create_app

        return create_app
    if name == "main":
        from mlxs.product_surfaces.cli import main

        return main
    raise AttributeError(name)
