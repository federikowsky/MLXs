"""Cache protocol — contract for KV cache implementations (§6.2, §9).

All cache types (full, quantized, rotating, batched) implement this protocol.
Consumers (generate, batch, prompt_cache) depend only on this interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx


@runtime_checkable
class CacheProtocol(Protocol):
    """Structural contract for KV caches.

    Each layer of the model holds one CacheProtocol instance. The cache
    accumulates key/value pairs across prefill and decode steps.
    """

    @property
    def offset(self) -> int:
        """Number of tokens currently stored in the cache."""
        ...

    @property
    def keys(self) -> mx.array | None:
        """Cached keys, or None if empty."""
        ...

    @property
    def values(self) -> mx.array | None:
        """Cached values, or None if empty."""
        ...

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Append new key/value pairs and return the full cached state.

        Args:
            keys: New keys ``(B, n_heads, T, head_dim)``.
            values: New values ``(B, n_heads, T, head_dim)``.

        Returns:
            Tuple of (all_keys, all_values) including the newly appended data.
        """
        ...

    def trim(self, n: int) -> int:
        """Remove the last ``n`` tokens from the cache (for rewind).

        Used by speculative decoding (§6.5) to rewind on rejection.

        Args:
            n: Number of tokens to remove from the end.

        Returns:
            Actual number of tokens removed (may be less if cache is shorter).
        """
        ...

    @property
    def state_size_bytes(self) -> int:
        """Approximate memory footprint of cached state in bytes.

        Used by prompt_cache for byte-based eviction limits (§6.3).
        """
        ...

    def reset(self) -> None:
        """Clear all cached state, resetting offset to 0."""
        ...
