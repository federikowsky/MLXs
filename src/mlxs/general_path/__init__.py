"""Layer 2 / General Path single-request generation semantics."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def generate_single_request(*args: Any, **kwargs: Any) -> Iterator[Any]:
    """Lazy wrapper so package import does not require MLX until execution."""
    from mlxs.general_path.single_request import generate_single_request as _impl

    return _impl(*args, **kwargs)


__all__ = ["generate_single_request"]
