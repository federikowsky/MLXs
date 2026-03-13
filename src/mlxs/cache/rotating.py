"""Rotating KV cache — bounded size with rotation (§6.2, FR4, AC4).

Implements a circular buffer that keeps the first ``keep`` tokens and
rotates the rest. Compatible with long-context generation where full
KV cache would exceed memory.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.cache.attention_mask import _mask_from_length


class RotatingKVCache:
    """Bounded rotating KV cache with configurable keep region.

    The cache has a maximum size. The first ``keep`` positions are
    always preserved (e.g. system prompt tokens). When the cache
    exceeds ``max_size``, older entries (after ``keep``) are evicted
    in a circular fashion.

    Satisfies ``CacheProtocol``.
    """

    step: int = 256

    __slots__ = ("_idx", "_keys", "_offset", "_values", "keep", "max_size")

    def __init__(self, max_size: int, keep: int = 0) -> None:
        self.keep = keep
        self.max_size = max_size
        self._keys: mx.array | None = None
        self._values: mx.array | None = None
        self._offset: int = 0
        self._idx: int = 0

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def keys(self) -> mx.array | None:
        if self._keys is None:
            return None
        if self._offset < self._keys.shape[2]:
            return self._keys[..., : self._offset, :]
        return self._keys

    @property
    def values(self) -> mx.array | None:
        if self._values is None:
            return None
        if self._offset < self._values.shape[2]:
            return self._values[..., : self._offset, :]
        return self._values

    def _trim_and_cat(
        self, v: mx.array, trim_size: int, append: mx.array | None = None
    ) -> mx.array:
        to_cat: list[mx.array] = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v: mx.array) -> mx.array:
        """Rearrange cache into temporal order."""
        if self._idx == v.shape[2]:
            return v
        if self._idx < self._offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        return v[..., : self._idx, :]

    def _update_concat(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Handle multi-token updates (prefill)."""
        if self._keys is None:
            self._keys = keys
            self._values = values
        else:
            self._keys = self._temporal_order(self._keys)
            self._values = self._temporal_order(self._values)
            self._idx = self._keys.shape[2]

            trim_size = self._idx - self.max_size + 1
            self._keys = self._trim_and_cat(trim_size, self._keys, keys)
            self._values = self._trim_and_cat(trim_size, self._values, values)

        self._offset += keys.shape[2]
        self._idx = self._keys.shape[2]
        return self._keys, self._values

    def _update_in_place(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Handle single-token updates (decode)."""
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        prev = self._offset

        if self._keys is None or (
            prev >= self._keys.shape[2] and self._keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self._keys is not None:
                self._keys = mx.concatenate([self._keys, new_k], axis=2)
                self._values = mx.concatenate([self._values, new_v], axis=2)
            else:
                self._keys, self._values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self._keys.shape[2] - self.max_size
        if trim_size > 0:
            self._keys = self._trim_and_cat(trim_size, self._keys)
            self._values = self._trim_and_cat(trim_size, self._values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self._keys[..., self._idx : self._idx + S, :] = keys
        self._values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self._idx += S

        # If buffer not full, slice off the end
        if self._offset < self.max_size:
            return (
                self._keys[..., : self._offset, :],
                self._values[..., : self._offset, :],
            )
        return self._keys, self._values

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def trim(self, n: int) -> int:
        n = min(self._offset, n)
        self._offset -= n
        return n

    @property
    def state(self) -> tuple[mx.array, mx.array] | None:
        if self._keys is None:
            return None
        if self._offset < self._keys.shape[2]:
            return (
                self._keys[..., : self._offset, :],
                self._values[..., : self._offset, :],
            )
        return self._keys, self._values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        self._keys, self._values = v

    @property
    def state_size_bytes(self) -> int:
        if self._keys is None:
            return 0
        return self._keys.nbytes + self._values.nbytes

    def reset(self) -> None:
        self._keys = None
        self._values = None
        self._offset = 0
        self._idx = 0

    def empty(self) -> bool:
        return self._keys is None

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        return _mask_from_length(
            n, offset=self._offset, return_array=return_array, window_size=window_size
        )
