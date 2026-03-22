"""Adaptive KV resident storage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

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


def _slice_quantized_state(
    state: QuantizedState,
    start: int,
    end: int,
) -> QuantizedState:
    return tree_map(lambda x: x[..., start:end, :], state)


def _concat_quantized_states(states: tuple[QuantizedState, ...]) -> QuantizedState:
    if len(states) == 1:
        return states[0]
    parts = list(zip(*states, strict=True))
    return tuple(  # type: ignore[return-value]
        mx.concatenate(list(component), axis=-2) for component in parts
    )


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


@dataclass(slots=True)
class AdaptiveCompressedRunStore:
    """One exact-sized compressed resident run spanning adjacent logical blocks."""

    run_id: int
    q_keys: QuantizedState
    q_values: QuantizedState
    group_size: int
    bits: int
    block_slices: tuple[tuple[int, int, int], ...]
    token_count: int

    @classmethod
    def from_full_block(
        cls,
        run_id: int,
        *,
        block_id: int,
        keys: mx.array,
        values: mx.array,
        group_size: int,
        bits: int,
    ) -> AdaptiveCompressedRunStore:
        q_keys = mx.quantize(keys, group_size=group_size, bits=bits)
        q_values = mx.quantize(values, group_size=group_size, bits=bits)
        token_count = keys.shape[2]
        return cls(
            run_id=run_id,
            q_keys=q_keys,
            q_values=q_values,
            group_size=group_size,
            bits=bits,
            block_slices=((block_id, 0, token_count),),
            token_count=token_count,
        )

    @property
    def live_bytes(self) -> int:
        return tree_reduce(lambda a, x: a + x.nbytes, (self.q_keys, self.q_values), 0)

    def state_tensors(self) -> tuple[QuantizedState, QuantizedState]:
        return self.q_keys, self.q_values

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

    def block_slice(self, block_id: int) -> tuple[int, int]:
        for current_block_id, start, end in self.block_slices:
            if current_block_id == block_id:
                return start, end
        raise KeyError(block_id)

    def block_live_bytes(self, block_id: int) -> int:
        start, end = self.block_slice(block_id)
        q_keys = _slice_quantized_state(self.q_keys, start, end)
        q_values = _slice_quantized_state(self.q_values, start, end)
        return tree_reduce(lambda a, x: a + x.nbytes, (q_keys, q_values), 0)

    def block_tensors(self, block_id: int, *, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        start, end = self.block_slice(block_id)
        q_keys = _slice_quantized_state(self.q_keys, start, end)
        q_values = _slice_quantized_state(self.q_values, start, end)
        keys = mx.dequantize(
            q_keys[0],
            q_keys[1],
            q_keys[2],
            group_size=self.group_size,
            bits=self.bits,
            dtype=dtype,
        )
        values = mx.dequantize(
            q_values[0],
            q_values[1],
            q_values[2],
            group_size=self.group_size,
            bits=self.bits,
            dtype=dtype,
        )
        return keys, values

    def merge_with(
        self,
        other: AdaptiveCompressedRunStore,
        *,
        run_id: int,
    ) -> AdaptiveCompressedRunStore:
        if (
            self.group_size != other.group_size
            or self.bits != other.bits
            or self.q_keys[0].shape[-1] != other.q_keys[0].shape[-1]
            or self.q_values[0].shape[-1] != other.q_values[0].shape[-1]
        ):
            raise ValueError("Compressed runs are not compatible for merge")
        shifted_block_slices = tuple(
            (block_id, start + self.token_count, end + self.token_count)
            for block_id, start, end in other.block_slices
        )
        return AdaptiveCompressedRunStore(
            run_id=run_id,
            q_keys=_concat_quantized_states((self.q_keys, other.q_keys)),
            q_values=_concat_quantized_states((self.q_values, other.q_values)),
            group_size=self.group_size,
            bits=self.bits,
            block_slices=self.block_slices + shifted_block_slices,
            token_count=self.token_count + other.token_count,
        )

    def split_without_block(
        self,
        block_id: int,
        *,
        left_run_id: int | None = None,
        right_run_id: int | None = None,
    ) -> tuple[tuple[QuantizedState, QuantizedState], tuple[AdaptiveCompressedRunStore, ...]]:
        block_index = -1
        block_start = 0
        block_end = 0
        for idx, (current_block_id, start, end) in enumerate(self.block_slices):
            if current_block_id == block_id:
                block_index = idx
                block_start = start
                block_end = end
                break
        if block_index < 0:
            raise KeyError(block_id)

        block_state = (
            _slice_quantized_state(self.q_keys, block_start, block_end),
            _slice_quantized_state(self.q_values, block_start, block_end),
        )
        fragments: list[AdaptiveCompressedRunStore] = []

        if block_start > 0:
            if left_run_id is None:
                raise ValueError("left_run_id is required for a left compressed fragment")
            fragments.append(
                AdaptiveCompressedRunStore(
                    run_id=left_run_id,
                    q_keys=_slice_quantized_state(self.q_keys, 0, block_start),
                    q_values=_slice_quantized_state(self.q_values, 0, block_start),
                    group_size=self.group_size,
                    bits=self.bits,
                    block_slices=self.block_slices[:block_index],
                    token_count=block_start,
                )
            )

        if block_end < self.token_count:
            if right_run_id is None:
                raise ValueError("right_run_id is required for a right compressed fragment")
            fragments.append(
                AdaptiveCompressedRunStore(
                    run_id=right_run_id,
                    q_keys=_slice_quantized_state(self.q_keys, block_end, self.token_count),
                    q_values=_slice_quantized_state(self.q_values, block_end, self.token_count),
                    group_size=self.group_size,
                    bits=self.bits,
                    block_slices=tuple(
                        (current_block_id, start - block_end, end - block_end)
                        for current_block_id, start, end in self.block_slices[block_index + 1 :]
                    ),
                    token_count=self.token_count - block_end,
                )
            )

        return block_state, tuple(fragments)
