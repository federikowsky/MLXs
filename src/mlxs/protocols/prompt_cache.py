"""Prompt cache protocol — contract for prefix KV cache reuse (§6.3, §9).

The prompt_cache module implements LRU caching of KV state by prefix.
Server and batch depend on this protocol for cache lookup/insert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mlxs.protocols.cache import CacheProtocol


@dataclass(frozen=True, slots=True)
class PromptCacheStats:
    """Snapshot of prompt cache statistics (§5.3)."""

    hit_count: int
    miss_count: int
    eviction_count: int
    entry_count: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class PromptCacheEntry:
    """A cached prefix and its KV state."""

    prefix_tokens: tuple[int, ...]
    cache_state: list[CacheProtocol]
    size_bytes: int


@runtime_checkable
class PromptCacheProtocol(Protocol):
    """Contract for the prompt prefix cache (§6.3, §9).

    Key = (model_id, prefix_token_ids). Lookup uses longest common prefix.
    """

    def get(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
        *,
        media_hash: str | None = None,
    ) -> tuple[list[CacheProtocol] | None, int]:
        """Look up the longest cached prefix for the given token sequence.

        Args:
            model_id: Identifier for the model (to avoid cross-model hits).
            token_ids: Full prompt token sequence.
            media_hash: Optional hash of media content (§7.4). When provided,
                only entries with the same media_hash are matched, preventing
                cache hits when different images share the same placeholder
                token sequence.

        Returns:
            Tuple of (cache_state, prefix_length):
            - cache_state: KV caches for the matched prefix, or None on miss.
            - prefix_length: Number of tokens matched (0 on miss).
        """
        ...

    def put(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
        cache_state: list[CacheProtocol],
        *,
        media_hash: str | None = None,
    ) -> None:
        """Store a prefix and its KV cache state.

        May trigger eviction if limits are exceeded. May trigger memory
        monitoring checks (§6.3.1).

        Args:
            model_id: Identifier for the model.
            token_ids: Prefix token sequence.
            cache_state: KV caches to store.
            media_hash: Optional hash of media content (§7.4). When provided,
                the entry is keyed by (model_id, token_ids, media_hash) to
                disambiguate identical token sequences with different media.
        """
        ...

    def trim(self, n: int) -> int:
        """Remove up to ``n`` least-recently-used entries.

        Args:
            n: Maximum entries to remove.

        Returns:
            Actual number of entries removed.
        """
        ...

    def clear(self) -> None:
        """Remove all entries from the cache."""
        ...

    def stats(self) -> PromptCacheStats:
        """Return current cache statistics."""
        ...
