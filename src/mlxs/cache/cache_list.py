"""CacheList — wrapper that holds multiple caches per layer (e.g. ArraysCache + KVCache).

Used by hybrid models (Falcon-H1, Jamba) where each layer has both an SSM/Mamba
cache and a KV attention cache. The layer uses cache[0] for SSM and cache[1]
for attention; generate uses the KV cache for offset/keys/values (CacheProtocol).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


class CacheList:
    """Container of N caches per layer; delegates CacheProtocol to the KV cache.

    Typically used as CacheList(ArraysCache(size=2), KVCache()) so the model
    can use cache[0] for SSM state and cache[1] for attention KV. offset, keys,
    values, and update_and_fetch are delegated to the cache at kv_index (default
    last) so that generate and mask creation see the KV cache.
    """

    def __init__(self, *caches: Any, kv_index: int = -1) -> None:
        if not caches:
            raise ValueError("CacheList requires at least one cache")
        self.caches = list(caches)
        self._kv_index = kv_index if kv_index >= 0 else len(self.caches) + kv_index

    def __getitem__(self, idx: int) -> Any:
        return self.caches[idx]

    @property
    def _kv(self) -> Any:
        return self.caches[self._kv_index]

    @property
    def offset(self) -> int:
        return getattr(self._kv, "offset", 0)

    @property
    def keys(self) -> mx.array | None:
        return getattr(self._kv, "keys", None)

    @property
    def values(self) -> mx.array | None:
        return getattr(self._kv, "values", None)

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return self._kv.update_and_fetch(keys, values)

    def trim(self, n: int) -> int:
        out = 0
        for c in self.caches:
            if hasattr(c, "trim"):
                out = c.trim(n)
        return out

    @property
    def state_size_bytes(self) -> int:
        total = 0
        for c in self.caches:
            total += getattr(c, "state_size_bytes", 0) or 0
        return total

    def reset(self) -> None:
        for c in self.caches:
            if hasattr(c, "reset"):
                c.reset()

    def make_mask(self, n: int, **kwargs: Any) -> mx.array | str | None:
        """Delegate to KV cache for attention mask (offset-aware)."""
        if hasattr(self._kv, "make_mask"):
            return self._kv.make_mask(n, **kwargs)
        return None
