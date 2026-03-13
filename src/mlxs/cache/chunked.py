"""Chunked KV cache for chunked attention (e.g. Llama 4).

Keeps at most chunk_size tokens per layer; used with chunk_mask for
local attention. Compatible with mlx_lm ChunkedKVCache.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.cache.attention_mask import _mask_from_length


class ChunkedKVCache:
    """KV cache that trims the front to stay within chunk_size (Llama 4 chunked layers).

    Satisfies CacheProtocol. Uses start_position for chunk-relative positions
    when building chunk masks. maybe_trim_front() keeps cache size <= chunk_size.
    """

    step: int = 256

    __slots__ = ("_chunk_size", "_keys", "_offset", "_start_position", "_values")

    def __init__(self, chunk_size: int) -> None:
        self._keys: mx.array | None = None
        self._values: mx.array | None = None
        self._offset: int = 0
        self._chunk_size: int = chunk_size
        self._start_position: int = 0

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def start_position(self) -> int:
        return self._start_position

    @property
    def keys(self) -> mx.array | None:
        if self._keys is None:
            return None
        return self._keys[..., : self._offset, :]

    @property
    def values(self) -> mx.array | None:
        if self._values is None:
            return None
        return self._values[..., : self._offset, :]

    def maybe_trim_front(self) -> None:
        """Keep cache length <= chunk_size by dropping oldest tokens (Llama 4)."""
        if self._keys is None or self._keys.shape[2] < self._chunk_size:
            return
        drop = self._keys.shape[2] - self._chunk_size
        self._start_position += drop
        self._keys = self._keys[..., -self._chunk_size :, :]
        self._values = self._values[..., -self._chunk_size :, :]

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        prev = self._offset - self._start_position
        n_new = keys.shape[2]

        if self._keys is None or (prev + n_new) > self._keys.shape[2]:
            self._grow(keys, values, prev, n_new)

        self._offset += n_new
        end = self._offset - self._start_position
        self._keys[..., prev:end, :] = keys
        self._values[..., prev:end, :] = values
        return self._keys[..., :end, :], self._values[..., :end, :]

    def _grow(
        self,
        keys: mx.array,
        values: mx.array,
        prev: int,
        n_new: int,
    ) -> None:
        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        n_steps = (self.step + n_new - 1) // self.step
        k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
        v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
        new_k = mx.zeros(k_shape, keys.dtype)
        new_v = mx.zeros(v_shape, values.dtype)
        if self._keys is not None:
            if prev % self.step != 0:
                self._keys = self._keys[..., :prev, :]
                self._values = self._values[..., :prev, :]
            self._keys = mx.concatenate([self._keys, new_k], axis=2)
            self._values = mx.concatenate([self._values, new_v], axis=2)
        else:
            self._keys, self._values = new_k, new_v

    def trim(self, n: int) -> int:
        n = min(self._offset - self._start_position, n)
        self._offset -= n
        return n

    @property
    def state_size_bytes(self) -> int:
        if self._keys is None:
            return 0
        return self._keys.nbytes + self._values.nbytes

    def reset(self) -> None:
        self._keys = None
        self._values = None
        self._offset = 0
        self._start_position = 0

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        """Causal mask for chunk-relative length (used when no chunk_mask)."""
        return _mask_from_length(
            n,
            offset=self._offset - self._start_position,
            return_array=return_array,
            window_size=window_size,
        )
