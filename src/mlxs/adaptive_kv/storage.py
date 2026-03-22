"""Adaptive KV resident storage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import mlx.core as mx
from mlx.utils import tree_reduce

from mlxs.adaptive_kv.block_types import BlockTier

QuantizedState: TypeAlias = tuple[mx.array, mx.array, mx.array]


@dataclass(frozen=True, slots=True)
class AdaptiveAttentionSegment:
    """One ordered resident attention segment."""

    tier: BlockTier
    token_count: int
    block_slices: tuple[tuple[int, int, int], ...]
    full_slice: tuple[int, int] | None = None
    q_keys: QuantizedState | None = None
    q_values: QuantizedState | None = None
    group_size: int | None = None
    bits: int | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveResidentState:
    """Ordered resident view consumed by adaptive attention."""

    total_tokens: int
    segments: tuple[AdaptiveAttentionSegment, ...]
    full_keys: mx.array | None = None
    full_values: mx.array | None = None

    @property
    def has_compressed(self) -> bool:
        return any(segment.tier is BlockTier.COMPRESSED for segment in self.segments)


@dataclass(slots=True)
class AdaptiveCompressedBlockStore:
    """Exact-sized quantized resident storage for one logical block."""

    q_keys: QuantizedState
    q_values: QuantizedState
    group_size: int
    bits: int
    token_count: int

    @classmethod
    def from_full(
        cls,
        keys: mx.array,
        values: mx.array,
        *,
        group_size: int,
        bits: int,
    ) -> AdaptiveCompressedBlockStore:
        q_keys = mx.quantize(keys, group_size=group_size, bits=bits)
        q_values = mx.quantize(values, group_size=group_size, bits=bits)
        return cls(
            q_keys=q_keys,
            q_values=q_values,
            group_size=group_size,
            bits=bits,
            token_count=keys.shape[2],
        )

    @property
    def live_bytes(self) -> int:
        return tree_reduce(lambda a, x: a + x.nbytes, (self.q_keys, self.q_values), 0)

    def dequantize(self, *, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        keys = mx.dequantize(
            self.q_keys[0],
            self.q_keys[1],
            self.q_keys[2],
            group_size=self.group_size,
            bits=self.bits,
            dtype=dtype,
        )
        values = mx.dequantize(
            self.q_values[0],
            self.q_values[1],
            self.q_values[2],
            group_size=self.group_size,
            bits=self.bits,
            dtype=dtype,
        )
        return keys, values

    def state_tensors(self) -> tuple[QuantizedState, QuantizedState]:
        return self.q_keys, self.q_values
