"""Arrays-based cache for non-KV models (Mamba, RWKV, etc.).

Used by state-space models that don't use traditional key/value caches
but instead maintain recurrent state arrays. Compatible with mlx_lm's
ArraysCache API.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


class ArraysCache:
    """List-based cache for state-space and recurrent models.

    Stores arbitrary state arrays indexed by position, with support
    for left-padding and sequence-length tracking for batched inference.
    """

    def __init__(
        self,
        size: int,
        left_padding: list[int] | None = None,
    ) -> None:
        self.cache: list[Any] = [None] * size
        self.left_padding: mx.array | None = (
            mx.array(left_padding) if left_padding else None
        )
        self.lengths: mx.array | None = None

    def __setitem__(self, idx: int, value: Any) -> None:
        self.cache[idx] = value

    def __getitem__(self, idx: int) -> Any:
        return self.cache[idx]

    @property
    def offset(self) -> int:
        return 0

    @property
    def keys(self) -> None:
        return None

    @property
    def values(self) -> None:
        return None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        raise NotImplementedError("ArraysCache does not support KV update_and_fetch")

    def trim(self, n: int) -> int:
        return 0

    @property
    def state_size_bytes(self) -> int:
        total = 0
        for c in self.cache:
            if c is not None and hasattr(c, "nbytes"):
                total += c.nbytes
        return total

    def reset(self) -> None:
        self.cache = [None] * len(self.cache)
        self.left_padding = None
        self.lengths = None

    def empty(self) -> bool:
        return all(c is None for c in self.cache)

    def make_mask(self, n: int) -> mx.array | None:
        """Create mask for batched SSM inference."""
        if self.left_padding is not None:
            pos = mx.arange(n)
            return pos >= self.left_padding[:, None]
        if self.lengths is not None:
            pos = mx.arange(n)
            return pos < self.lengths[:, None]
        return None

    def prepare(self, lengths: list[int] | None = None, **kwargs: Any) -> None:
        if lengths is not None:
            self.lengths = mx.array(lengths)

    def finalize(self) -> None:
        self.lengths = None
        self.left_padding = None

    def advance(self, n: int) -> None:
        if self.lengths is not None:
            self.lengths -= n
        if self.left_padding is not None:
            self.left_padding -= n
