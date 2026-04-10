"""Compatibility wrapper over the Layer 4 product-runtime bootstrap."""

from mlxs.product_surfaces.bootstrap import ProductRuntime as Dependencies
from mlxs.product_surfaces.bootstrap import create_runtime


def create_dependencies(config):  # type: ignore[no-untyped-def]
    """Compatibility wrapper delegating to the sole Layer 4 composition root."""
    return create_runtime(config)


__all__ = ["Dependencies", "create_dependencies"]
