"""Full-precision KV cache (§6.2, FR4, AC4).

Implements CacheProtocol. Pre-allocates in chunks to avoid per-token
allocation (O2). Compatible with mlx_lm cache shapes and patterns.
"""

from __future__ import annotations

from typing import cast

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

    @offset.setter
    def offset(self, value: int) -> None:
        if value < 0:
            raise ValueError("KVCache.offset must be >= 0")
        if self._keys is None:
            if value != 0:
                raise ValueError("Cannot set KVCache.offset on an empty cache")
            self._offset = 0
            return
        if value > self._keys.shape[2]:
            raise ValueError(
                f"KVCache.offset {value} exceeds allocated capacity {self._keys.shape[2]}"
            )
        self._offset = value

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

    @property
    def tracked_arrays(self) -> tuple[mx.array, mx.array]:
        if self._keys is None or self._values is None:
            raise ValueError("KVCache tracked arrays are unavailable before allocation")
        return self._keys, self._values

    def ensure_total_slots(self, total_slots: int) -> None:
        """Ensure the underlying buffers can hold ``total_slots`` entries.

        This is intended for compile-path pre-allocation before the first trace,
        so later decode steps do not trigger ``_grow()`` and invalidate tracked
        array identities.
        """
        if total_slots < 0:
            raise ValueError("total_slots must be >= 0")
        if total_slots == 0:
            return
        if self._keys is None or self._values is None:
            raise ValueError("Cannot pre-allocate KVCache before initial allocation")

        current_capacity = self._keys.shape[2]
        if current_capacity >= total_slots:
            return

        slots_needed = total_slots - current_capacity
        n_steps = (slots_needed + self.step - 1) // self.step
        extra_slots = n_steps * self.step
        k_shape = (*self._keys.shape[:2], extra_slots, self._keys.shape[3])
        v_shape = (*self._values.shape[:2], extra_slots, self._values.shape[3])
        extra_k = mx.zeros(k_shape, dtype=self._keys.dtype)
        extra_v = mx.zeros(v_shape, dtype=self._values.dtype)
        self._keys = mx.concatenate([self._keys, extra_k], axis=2)
        self._values = mx.concatenate([self._values, extra_v], axis=2)

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        prev = self._offset
        n_new = keys.shape[2]

        if self._keys is None or (prev + n_new) > self._keys.shape[2]:
            self._grow(keys, values, prev, n_new)

        keys_buf = cast(mx.array, self._keys)
        values_buf = cast(mx.array, self._values)
        self._offset = prev + n_new
        keys_buf[..., prev : self._offset, :] = keys
        values_buf[..., prev : self._offset, :] = values
        return keys_buf[..., : self._offset, :], values_buf[..., : self._offset, :]

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
            values_buf = cast(mx.array, self._values)
            if prev % self.step != 0:
                self._keys = self._keys[..., :prev, :]
                self._values = values_buf[..., :prev, :]
                values_buf = self._values
            self._keys = mx.concatenate([self._keys, new_k], axis=2)
            self._values = mx.concatenate([values_buf, new_v], axis=2)
        else:
            self._keys, self._values = new_k, new_v

    def trim(self, n: int) -> int:
        n = min(self._offset, n)
        self._offset -= n
        return n

    @property
    def state(self) -> tuple[mx.array, mx.array] | None:
        """Return current state for serialization / prompt cache."""
        if self._keys is None:
            return None
        values_buf = cast(mx.array, self._values)
        if self._offset == self._keys.shape[2]:
            return self._keys, values_buf
        return self._keys[..., : self._offset, :], values_buf[..., : self._offset, :]

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        self._keys, self._values = v
        self._offset = self._keys.shape[2]

    @property
    def state_size_bytes(self) -> int:
        if self._keys is None:
            return 0
        return self._keys.nbytes + cast(mx.array, self._values).nbytes

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
        return cast(
            mx.array | str | None,
            _mask_from_length(
                n, offset=self._offset, return_array=return_array, window_size=window_size
            ),
        )
