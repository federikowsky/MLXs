"""Compatibility CLI entrypoint delegating to the Layer 4 surface."""

from mlxs.product_surfaces.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
