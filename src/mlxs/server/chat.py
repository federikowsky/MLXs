"""Layer 4 compatibility wrapper for the canonical product-surface chat module."""

from mlxs.product_surfaces.chat import (
    _handle_plain_command,
    run_chat_loop,
)

__all__ = ["run_chat_loop", "_handle_plain_command"]
