"""LRU prompt cache with longest-prefix matching (§6.3, FR5, AC5).

Stores KV cache states keyed by (model_id, prefix_tokens). Lookup
uses a trie for longest common prefix matching. Eviction is LRU by
entry count and optional byte limit.

Event-driven: no background threads or timers (§6.3.1).
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from mlxs.protocols.prompt_cache import PromptCacheStats

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CacheEntry:
    """Internal cache entry with metadata."""

    model_id: str
    prefix_tokens: tuple[int, ...]
    cache_state: list[Any]  # list[CacheProtocol]
    size_bytes: int
    access_count: int = 0


class _PrefixTrie:
    """Trie for longest-prefix lookup of token sequences.

    Each node stores a dict of model_id → entry_key to support
    multiple models sharing the same token prefix. Lookup walks the
    trie and returns the deepest node with a stored key for the
    requested model_id.
    """

    __slots__ = ("children", "entry_keys")

    def __init__(self) -> None:
        self.children: dict[int, _PrefixTrie] = {}
        self.entry_keys: dict[str, tuple[str, tuple[int, ...]]] = {}

    def insert(self, model_id: str, tokens: tuple[int, ...]) -> None:
        node = self
        for token in tokens:
            if token not in node.children:
                node.children[token] = _PrefixTrie()
            node = node.children[token]
        node.entry_keys[model_id] = (model_id, tokens)

    def longest_prefix(
        self, model_id: str, tokens: tuple[int, ...]
    ) -> tuple[tuple[str, tuple[int, ...]] | None, int]:
        """Find the longest stored prefix matching the start of tokens.

        Returns:
            (entry_key, prefix_length) — entry_key is None if no match.
        """
        node = self
        best_key: tuple[str, tuple[int, ...]] | None = None
        best_len = 0

        for i, token in enumerate(tokens):
            if token not in node.children:
                break
            node = node.children[token]
            if model_id in node.entry_keys:
                best_key = node.entry_keys[model_id]
                best_len = i + 1

        return best_key, best_len

    def remove(self, model_id: str, tokens: tuple[int, ...]) -> None:
        """Remove an entry for a specific model from the trie."""
        node = self
        for token in tokens:
            if token not in node.children:
                return
            node = node.children[token]
        node.entry_keys.pop(model_id, None)


class LRUPromptCache:
    """LRU prompt cache with prefix trie lookup.

    Satisfies ``PromptCacheProtocol``.

    Limits:
    - ``max_entries``: Maximum number of cached prefixes.
    - ``max_bytes``: Maximum total bytes of cached KV state (optional).

    Eviction: LRU — least recently accessed entries are evicted first.
    """

    __slots__ = (
        "_entries",
        "_eviction_count",
        "_hit_count",
        "_miss_count",
        "_total_bytes",
        "_trie",
        "max_bytes",
        "max_entries",
    )

    def __init__(
        self,
        *,
        max_entries: int = 100,
        max_bytes: int | None = None,
    ) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        # OrderedDict for LRU ordering: most recent at end
        self._entries: OrderedDict[tuple[str, tuple[int, ...]], _CacheEntry] = OrderedDict()
        self._trie = _PrefixTrie()
        self._total_bytes: int = 0
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._eviction_count: int = 0

    def get(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
    ) -> tuple[list[Any] | None, int]:
        """Look up the longest cached prefix.

        Returns a deep copy of the cache state so modifications by the
        caller don't corrupt the cached entry.
        """
        entry_key, prefix_len = self._trie.longest_prefix(model_id, token_ids)

        if entry_key is None or entry_key not in self._entries:
            self._miss_count += 1
            return None, 0

        entry = self._entries[entry_key]
        entry.access_count += 1
        # Move to end (most recently used)
        self._entries.move_to_end(entry_key)
        self._hit_count += 1

        # Deep copy cache state so caller can mutate freely
        cache_copy = copy.deepcopy(entry.cache_state)
        return cache_copy, prefix_len

    def put(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
        cache_state: list[Any],
    ) -> None:
        """Store a prefix and its KV cache state.

        Evicts LRU entries if limits are exceeded.
        """
        key = (model_id, token_ids)

        # Calculate size
        size_bytes = sum(getattr(c, "state_size_bytes", 0) for c in cache_state)

        # If entry already exists, update it
        if key in self._entries:
            old_entry = self._entries[key]
            self._total_bytes -= old_entry.size_bytes
            old_entry.cache_state = copy.deepcopy(cache_state)
            old_entry.size_bytes = size_bytes
            self._total_bytes += size_bytes
            self._entries.move_to_end(key)
            return

        # Store deep copy
        entry = _CacheEntry(
            model_id=model_id,
            prefix_tokens=token_ids,
            cache_state=copy.deepcopy(cache_state),
            size_bytes=size_bytes,
        )

        self._entries[key] = entry
        self._trie.insert(model_id, token_ids)
        self._total_bytes += size_bytes

        # Evict if over count limit
        while len(self._entries) > self.max_entries:
            self._evict_lru()

        # Evict if over byte limit
        if self.max_bytes is not None:
            while self._total_bytes > self.max_bytes and self._entries:
                self._evict_lru()

    def trim(self, n: int) -> int:
        """Remove up to n least-recently-used entries."""
        removed = 0
        for _ in range(n):
            if not self._entries:
                break
            self._evict_lru()
            removed += 1
        return removed

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._trie = _PrefixTrie()
        self._total_bytes = 0

    def stats(self) -> PromptCacheStats:
        return PromptCacheStats(
            hit_count=self._hit_count,
            miss_count=self._miss_count,
            eviction_count=self._eviction_count,
            entry_count=len(self._entries),
            total_bytes=self._total_bytes,
        )

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._entries:
            return
        _key, entry = self._entries.popitem(last=False)
        self._trie.remove(entry.model_id, entry.prefix_tokens)
        self._total_bytes -= entry.size_bytes
        self._eviction_count += 1
        logger.debug(
            "Evicted prompt cache entry: model=%s, prefix_len=%d, bytes=%d",
            entry.model_id,
            len(entry.prefix_tokens),
            entry.size_bytes,
        )
