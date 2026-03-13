"""Quantized KV cache — reduced memory via quantization (§6.2, FR4, AC4).

Stores keys and values in quantized format (configurable bits and group_size).
Compatible with mlx.fast.scaled_dot_product_attention quantized path.
Uses the same chunked pre-allocation strategy as KVCache.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

from mlxs.cache.attention_mask import create_attention_mask


class QuantizedKVCache:
    """Quantized KV cache with chunked pre-allocation.

    Keys and values are quantized on insert using ``mx.quantize()``.
    Each stored entry is a tuple of (data, scales, biases).

    Satisfies ``CacheProtocol``.
    """

    step: int = 256

    __slots__ = ("_idx", "_offset", "bits", "group_size", "keys", "values")

    def __init__(self, group_size: int = 64, bits: int = 8) -> None:
        self.keys: tuple[mx.array, mx.array, mx.array] | None = None
        self.values: tuple[mx.array, mx.array, mx.array] | None = None
        self._offset: int = 0
        self.group_size = group_size
        self.bits = bits

    @property
    def offset(self) -> int:
        return self._offset

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[
        tuple[mx.array, mx.array, mx.array],
        tuple[mx.array, mx.array, mx.array],
    ]:
        """Update cache with new keys/values, quantizing on insert."""
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def _init_quant(dim: int) -> tuple[mx.array, mx.array, mx.array]:
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def _expand(x: mx.array) -> mx.array:
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                self.keys, self.values = tree_map(_expand, (self.keys, self.values))
            else:
                self.keys = _init_quant(k_head_dim)
                self.values = _init_quant(v_head_dim)

        self._offset += num_steps

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self._offset, :] = q_keys[i]
            self.values[i][..., prev : self._offset, :] = q_values[i]

        return (
            tree_map(lambda x: x[..., : self._offset, :], self.keys),
            tree_map(lambda x: x[..., : self._offset, :], self.values),
        )

    def trim(self, n: int) -> int:
        n = min(self._offset, n)
        self._offset -= n
        return n

    @property
    def state(self) -> tuple | None:
        if self.keys is None:
            return None
        if self._offset == self.keys[0].shape[2]:
            return self.keys, self.values
        return tree_map(lambda x: x[..., : self._offset, :], (self.keys, self.values))

    @state.setter
    def state(self, v: tuple) -> None:
        self.keys, self.values = v

    @property
    def state_size_bytes(self) -> int:
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self._offset = 0

    def empty(self) -> bool:
        return self.keys is None

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array | str | None:
        return create_attention_mask(
            n, offset=self._offset, return_array=return_array, window_size=window_size
        )
