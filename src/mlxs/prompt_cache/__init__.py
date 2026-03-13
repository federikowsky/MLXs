"""Prompt cache module — LRU prefix cache with memory monitoring (§6.3, §9).

Public API: ``PromptCache`` — wraps LRU store with memory-aware insert.
"""

from __future__ import annotations

import logging
from typing import Any

from mlxs.config.schema import PromptCacheConfig
from mlxs.prompt_cache.lru import LRUPromptCache
from mlxs.prompt_cache.memory import MemoryAction, MemoryMonitor
from mlxs.protocols.prompt_cache import PromptCacheStats

logger = logging.getLogger(__name__)


class PromptCache:
    """Memory-aware prompt cache (§6.3, FR5, AC5).

    Wraps ``LRUPromptCache`` with event-driven memory monitoring.
    Memory checks fire on ``put()`` — no background threads (§6.3.1).

    Satisfies ``PromptCacheProtocol``.
    """

    __slots__ = ("_lru", "_monitor")

    def __init__(
        self,
        config: PromptCacheConfig | None = None,
        *,
        monitor: MemoryMonitor | None = None,
    ) -> None:
        if config is None:
            config = PromptCacheConfig()

        self._lru = LRUPromptCache(
            max_entries=config.max_entries,
            max_bytes=config.max_bytes,
        )

        self._monitor = monitor or MemoryMonitor(
            max_rss_bytes=(
                int(config.trim_on_rss_gb * 1024**3) if config.trim_on_rss_gb is not None else None
            ),
            target_rss_ratio=config.target_rss_ratio,
            policy=config.on_memory_ceiling,
            trim_step=config.trim_step,
        )

    def get(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
    ) -> tuple[list[Any] | None, int]:
        """Look up the longest cached prefix."""
        return self._lru.get(model_id, token_ids)

    def put(
        self,
        model_id: str,
        token_ids: tuple[int, ...],
        cache_state: list[Any],
    ) -> None:
        """Store a prefix. Triggers memory check after insert (§6.3.1)."""
        self._lru.put(model_id, token_ids, cache_state)

        # Event-driven memory check
        action = self._monitor.check()
        if action == MemoryAction.TRIM:
            logger.info("Memory pressure detected — trimming prompt cache")
            self._monitor.apply_trim(self._lru.trim)
        elif action == MemoryAction.REJECT:
            logger.warning("Memory ceiling reached — new requests may be rejected")

    def trim(self, n: int) -> int:
        """Remove up to n LRU entries."""
        return self._lru.trim(n)

    def clear(self) -> None:
        """Remove all entries."""
        self._lru.clear()

    def stats(self) -> PromptCacheStats:
        """Return current cache statistics."""
        return self._lru.stats()


__all__ = ["PromptCache"]
