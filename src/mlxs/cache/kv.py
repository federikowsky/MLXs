"""Full-precision KV cache (§6.2, FR4, AC4).

Implements CacheProtocol. Pre-allocates in chunks to avoid per-token
allocation (O2). Compatible with mlx_lm cache shapes and patterns.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.cache.attention_mask import _mask_from_length


class KVCache:
    """Unbounded full-precision KV cache with chunked pre-allocation.

    Pre-allocates in blocks of ``step`` tokens to minimize allocations
    in the decode loop (O2, §6.8 buffer reuse).

    Satisfies ``CacheProtocol``.
    """

    step: int = 256

    __slots__ = ("_keys", "_offset", "_values")

    def __init__(self) -> None:
        self._keys: mx.array | None = None
        self._values: mx.array | None = None
        self._offset: int = 0

    @property
    def offset(self) -> int:
        return self._offset

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

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        prev = self._offset
        n_new = keys.shape[2]

        if self._keys is None or (prev + n_new) > self._keys.shape[2]:
            self._grow(keys, values, prev, n_new)

        self._offset = prev + n_new
        self._keys[..., prev : self._offset, :] = keys
        self._values[..., prev : self._offset, :] = values
        return self._keys[..., : self._offset, :], self._values[..., : self._offset, :]

    def _grow(
        self,
        keys: mx.array,
        values: mx.array,
        prev: int,
        n_new: int,
    ) -> None:
        """Expand the pre-allocated buffer in chunks of ``step``."""
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
        n = min(self._offset, n)
        self._offset -= n
        return n

    def copy_token_range(self, start: int, end: int) -> tuple[mx.array, mx.array]:
        """Return a resident token slice in ``[start, end)``."""
        if self._keys is None or self._values is None:
            raise ValueError("Cannot copy from an empty KVCache")
        start = max(0, min(start, self._offset))
        end = max(start, min(end, self._offset))
        return (
            self._keys[..., start:end, :],
            self._values[..., start:end, :],
        )

    def remove_token_range(self, start: int, end: int) -> int:
        """Physically remove resident tokens in ``[start, end)``."""
        if self._keys is None or self._values is None:
            return 0
        start = max(0, min(start, self._offset))
        end = max(start, min(end, self._offset))
        removed = end - start
        if removed == 0:
            return 0
        keep_keys: list[mx.array] = []
        keep_values: list[mx.array] = []
        if start > 0:
            keep_keys.append(self._keys[..., :start, :])
            keep_values.append(self._values[..., :start, :])
        if end < self._offset:
            keep_keys.append(self._keys[..., end:self._offset, :])
            keep_values.append(self._values[..., end:self._offset, :])
        if not keep_keys:
            self.reset()
            return removed
        self._keys = keep_keys[0] if len(keep_keys) == 1 else mx.concatenate(keep_keys, axis=2)
        self._values = (
            keep_values[0]
            if len(keep_values) == 1
            else mx.concatenate(keep_values, axis=2)
        )
        self._offset -= removed
        return removed

    @property
    def state(self) -> tuple[mx.array, mx.array] | None:
        """Return current state for serialization / prompt cache."""
        if self._keys is None:
            return None
        if self._offset == self._keys.shape[2]:
            return self._keys, self._values
        return self._keys[..., : self._offset, :], self._values[..., : self._offset, :]

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        self._keys, self._values = v
        self._offset = self._keys.shape[2]

    @property
    def state_size_bytes(self) -> int:
        if self._keys is None:
            return 0
        return self._keys.nbytes + self._values.nbytes

    @property
    def live_state_size_bytes(self) -> int:
        """Exact bytes for the currently live resident prefix."""
        if self._keys is None or self._values is None:
            return 0
        keys = self.keys
        values = self.values
        if keys is None or values is None:
            return 0
        return keys.nbytes + values.nbytes

    def reset(self) -> None:
        self._keys = None
        self._values = None
        self._offset = 0

    def empty(self) -> bool:
        return self._keys is None

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        """Create attention mask compatible with mlx.fast.scaled_dot_product_attention."""
        return _mask_from_length(
            n, offset=self._offset, return_array=return_array, window_size=window_size
        )
