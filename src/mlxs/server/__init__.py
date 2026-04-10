"""Layer 4 compatibility exports over the canonical product-surface boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlxs.server.app import create_app
    from mlxs.server.deps import Dependencies, create_dependencies

__all__ = ["Dependencies", "create_app", "create_dependencies"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from mlxs.server.app import create_app

        return create_app
    if name in {"Dependencies", "create_dependencies"}:
        from mlxs.server.deps import Dependencies, create_dependencies

        return {
            "Dependencies": Dependencies,
            "create_dependencies": create_dependencies,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
