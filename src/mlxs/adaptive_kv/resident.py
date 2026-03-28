"""TurboQuant-first resident backend contracts and execution-fabric types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import mlx.core as mx

from mlxs.adaptive_kv.block_types import ResidentProfile


class ResidentExecutionMode(StrEnum):
    DIRECT_COMPRESSED = "direct_compressed"
    DEQUANTIZE_ON_READ = "dequantize_on_read"
    FAMILY_SPECIFIC = "family_specific"


@dataclass(frozen=True, slots=True)
class ResidentProfileDescriptor:
    profile: ResidentProfile
    slab_capacity_tokens: int
    execution_mode: ResidentExecutionMode
    fidelity_rank: int
    memory_cost_class: int


@dataclass(frozen=True, slots=True)
class ExecutionFragment:
    slab_id: int
    local_start: int
    local_end: int

    @property
    def token_count(self) -> int:
        return self.local_end - self.local_start


@dataclass(slots=True)
class ResidentBlockHandle:
    block_id: int
    profile: ResidentProfile
    logical_span: tuple[int, int]
    fragments: tuple[ExecutionFragment, ...]
    dtype: mx.Dtype
    execution_mode: ResidentExecutionMode
    slab_capacity_tokens: int
    per_token_live_bytes: int
    live_bytes: int

    @property
    def token_count(self) -> int:
        return self.logical_span[1] - self.logical_span[0]


@dataclass(slots=True)
class SlabBlockRange:
    block_id: int
    local_start: int
    local_end: int
    logical_start: int
    logical_end: int

    @property
    def token_count(self) -> int:
        return self.local_end - self.local_start


@dataclass(slots=True)
class ExecutionSlab:
    slab_id: int
    profile: ResidentProfile
    logical_span: tuple[int, int]
    keys: mx.array
    values: mx.array
    capacity_tokens: int
    used_tokens: int
    dtype: mx.Dtype
    tombstoned_ranges: list[tuple[int, int]] = field(default_factory=list)
    block_ranges: list[SlabBlockRange] = field(default_factory=list)

    @property
    def live_tokens(self) -> int:
        return sum(r.token_count for r in self.block_ranges)


@runtime_checkable
class ResidentBackend(Protocol):
    def descriptor(self, profile: ResidentProfile) -> ResidentProfileDescriptor: ...

    def create_handle(
        self,
        *,
        block_id: int,
        profile: ResidentProfile,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
    ) -> ResidentBlockHandle: ...

    def append_tokens(
        self,
        handle: ResidentBlockHandle,
        *,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
        dtype: mx.Dtype,
    ) -> ResidentBlockHandle: ...

    def convert_profile(
        self,
        handle: ResidentBlockHandle,
        *,
        profile: ResidentProfile,
        dtype: mx.Dtype,
    ) -> ResidentBlockHandle: ...

    def evict_handle(self, handle: ResidentBlockHandle) -> None: ...

    def materialize(self, handle: ResidentBlockHandle) -> tuple[mx.array, mx.array]: ...


def _concat_token_axis(arrays: list[mx.array]) -> mx.array:
    if not arrays:
        raise ValueError("Expected at least one tensor to concatenate")
    return arrays[0] if len(arrays) == 1 else mx.concatenate(arrays, axis=2)


def _per_token_live_bytes(keys: mx.array, values: mx.array) -> int:
    if keys.shape[2] <= 0 or values.shape[2] <= 0:
        return 0
    return int(keys[..., :1, :].nbytes + values[..., :1, :].nbytes)


def _slab_compatible(
    slab: ExecutionSlab,
    *,
    profile: ResidentProfile,
    dtype: mx.Dtype,
    keys: mx.array,
    values: mx.array,
) -> bool:
    return (
        slab.profile is profile
        and slab.dtype == dtype
        and slab.keys.shape[:2] == keys.shape[:2]
        and slab.keys.shape[-1] == keys.shape[-1]
        and slab.values.shape[:2] == values.shape[:2]
        and slab.values.shape[-1] == values.shape[-1]
    )


def _piece_bytes(keys: mx.array, values: mx.array) -> int:
    return int(keys.nbytes + values.nbytes)


@dataclass(frozen=True, slots=True)
class _VisiblePiece:
    block_id: int
    slab_id: int
    local_start: int
    local_end: int
    logical_start: int
    logical_end: int
    visible_logical_start: int
    visible_logical_end: int


class ResidentExecutionFabric:
    """Persistent exact execution substrate for resident KV state."""

    def __init__(self) -> None:
        self._slabs: dict[int, ExecutionSlab] = {}
        self._handles: dict[int, ResidentBlockHandle] = {}
        self._next_slab_id = 0
        self._compactions_total = 0

    @property
    def compactions_total(self) -> int:
        return self._compactions_total

    def register_handle(self, handle: ResidentBlockHandle) -> None:
        self._handles[handle.block_id] = handle

    def unregister_handle(self, block_id: int) -> None:
        self._handles.pop(block_id, None)

    def reset(self) -> None:
        self._slabs = {}
        self._handles = {}
        self._next_slab_id = 0
        self._compactions_total = 0

    def allocate_block(
        self,
        handle: ResidentBlockHandle,
        *,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
        allow_tail_fast: bool,
    ) -> None:
        token_count = logical_span[1] - logical_span[0]
        capacity = max(handle.slab_capacity_tokens, token_count)
        if allow_tail_fast:
            tail = self._find_tail_slab(
                profile=handle.profile,
                dtype=handle.dtype,
                keys=keys,
                values=values,
                logical_span=logical_span,
                capacity=capacity,
            )
            if tail is not None:
                self._append_new_block_to_slab(
                    tail,
                    handle=handle,
                    logical_span=logical_span,
                    keys=keys,
                    values=values,
                )
                return

        slab = self._find_insert_slab(
            profile=handle.profile,
            dtype=handle.dtype,
            keys=keys,
            values=values,
            logical_span=logical_span,
            capacity=capacity,
        )
        if slab is None:
            slab = self._new_slab(
                profile=handle.profile,
                dtype=handle.dtype,
                capacity=max(capacity, token_count),
                logical_span=logical_span,
                keys=keys,
                values=values,
            )
            block_range = SlabBlockRange(
                block_id=handle.block_id,
                local_start=0,
                local_end=token_count,
                logical_start=logical_span[0],
                logical_end=logical_span[1],
            )
            slab.block_ranges = [block_range]
            handle.fragments = (ExecutionFragment(slab.slab_id, 0, token_count),)
            handle.logical_span = logical_span
            handle.live_bytes = handle.token_count * handle.per_token_live_bytes
            return

        self._insert_block_into_existing_slab(
            slab,
            handle=handle,
            logical_span=logical_span,
            keys=keys,
            values=values,
        )

    def append_to_tail(
        self,
        handle: ResidentBlockHandle,
        *,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
    ) -> bool:
        if not handle.fragments:
            self.allocate_block(
                handle,
                logical_span=logical_span,
                keys=keys,
                values=values,
                allow_tail_fast=True,
            )
            return True

        last_fragment = handle.fragments[-1]
        slab = self._slabs[last_fragment.slab_id]
        token_count = keys.shape[2]
        can_extend_tail = (
            _slab_compatible(
                slab,
                profile=handle.profile,
                dtype=handle.dtype,
                keys=keys,
                values=values,
            )
            and slab.block_ranges
            and slab.block_ranges[-1].block_id == handle.block_id
            and slab.block_ranges[-1].local_end == slab.used_tokens
            and last_fragment.local_end == slab.used_tokens
        )
        if can_extend_tail:
            capacity = max(handle.slab_capacity_tokens, slab.capacity_tokens)
            if slab.used_tokens + token_count > slab.capacity_tokens:
                dedicated_single_block = (
                    len(slab.block_ranges) == 1
                    and slab.block_ranges[0].block_id == handle.block_id
                )
                if dedicated_single_block:
                    capacity = slab.used_tokens + token_count
                    slab.capacity_tokens = capacity
                else:
                    can_extend_tail = False
        if not can_extend_tail:
            self.allocate_block(
                handle,
                logical_span=(logical_span[1] - token_count, logical_span[1]),
                keys=keys,
                values=values,
                allow_tail_fast=True,
            )
            handle.logical_span = logical_span
            handle.live_bytes = handle.token_count * handle.per_token_live_bytes
            return True

        slab.keys = _concat_token_axis([slab.keys, keys.astype(handle.dtype)])
        slab.values = _concat_token_axis([slab.values, values.astype(handle.dtype)])
        slab.used_tokens += token_count
        slab.block_ranges[-1].local_end += token_count
        slab.block_ranges[-1].logical_end += token_count
        slab.logical_span = (slab.logical_span[0], slab.block_ranges[-1].logical_end)
        handle.fragments = (*handle.fragments[:-1], ExecutionFragment(
            slab.slab_id,
            last_fragment.local_start,
            last_fragment.local_end + token_count,
        ))
        handle.logical_span = logical_span
        handle.live_bytes = handle.token_count * handle.per_token_live_bytes
        return False

    def move_block(
        self,
        handle: ResidentBlockHandle,
        *,
        profile: ResidentProfile,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
        execution_mode: ResidentExecutionMode,
        slab_capacity_tokens: int,
    ) -> None:
        self._remove_handle_fragments(handle)
        handle.profile = profile
        handle.execution_mode = execution_mode
        handle.slab_capacity_tokens = slab_capacity_tokens
        handle.fragments = ()
        self.allocate_block(
            handle,
            logical_span=logical_span,
            keys=keys.astype(handle.dtype),
            values=values.astype(handle.dtype),
            allow_tail_fast=False,
        )
        handle.logical_span = logical_span
        handle.live_bytes = handle.token_count * handle.per_token_live_bytes

    def evict_block(self, handle: ResidentBlockHandle) -> None:
        self._remove_handle_fragments(handle)
        self.unregister_handle(handle.block_id)

    def materialize(self, handle: ResidentBlockHandle) -> tuple[mx.array, mx.array]:
        key_parts: list[mx.array] = []
        value_parts: list[mx.array] = []
        for fragment in handle.fragments:
            slab = self._slabs[fragment.slab_id]
            key_parts.append(slab.keys[..., fragment.local_start : fragment.local_end, :])
            value_parts.append(slab.values[..., fragment.local_start : fragment.local_end, :])
        if not key_parts:
            raise ValueError(f"Resident block {handle.block_id} has no execution fragments")
        return _concat_token_axis(key_parts), _concat_token_axis(value_parts)

    def debug_materialize(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
    ) -> tuple[mx.array | None, mx.array | None]:
        if not ordered_handles:
            return None, None
        keys_list: list[mx.array] = []
        values_list: list[mx.array] = []
        for handle in ordered_handles:
            keys, values = self.materialize(handle)
            keys_list.append(keys)
            values_list.append(values)
        return _concat_token_axis(keys_list), _concat_token_axis(values_list)

    def query_full_view(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
        *,
        topology_epoch: int,
        tail_epoch: int,
        execution_view_topology_rebuilds_total: int,
    ) -> Any:
        return self._query_view(
            ordered_handles,
            visible_start=0,
            topology_epoch=topology_epoch,
            tail_epoch=tail_epoch,
            execution_view_topology_rebuilds_total=execution_view_topology_rebuilds_total,
        )

    def query_window_view(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
        *,
        logical_offset: int,
        query_tokens: int,
        window_size: int,
        topology_epoch: int,
        tail_epoch: int,
        execution_view_topology_rebuilds_total: int,
    ) -> Any:
        visible_start = max(0, logical_offset - query_tokens - max(0, window_size - 1))
        return self._query_view(
            ordered_handles,
            visible_start=visible_start,
            topology_epoch=topology_epoch,
            tail_epoch=tail_epoch,
            execution_view_topology_rebuilds_total=execution_view_topology_rebuilds_total,
        )

    def slabs_for_handles(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
    ) -> tuple[ExecutionSlab, ...]:
        seen: dict[int, ExecutionSlab] = {}
        for handle in ordered_handles:
            for fragment in handle.fragments:
                slab = self._slabs.get(fragment.slab_id)
                if slab is not None and fragment.slab_id not in seen:
                    seen[fragment.slab_id] = slab
        return tuple(seen.values())

    def _query_view(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
        *,
        visible_start: int,
        topology_epoch: int,
        tail_epoch: int,
        execution_view_topology_rebuilds_total: int,
    ) -> Any:
        from mlxs.adaptive_kv.storage import ExecutionSliceRef, ResidentStateView

        pieces = self._visible_pieces(ordered_handles, visible_start=visible_start)
        slices: list[ExecutionSliceRef] = []
        resident_cursor = 0
        slab_token_counts: dict[int, int] = {}

        idx = 0
        while idx < len(pieces):
            first = pieces[idx]
            slab = self._slabs[first.slab_id]
            group = [first]
            idx += 1
            while idx < len(pieces):
                candidate = pieces[idx]
                prev = group[-1]
                if (
                    candidate.slab_id == prev.slab_id
                    and candidate.local_start == prev.local_end
                    and candidate.visible_logical_start == prev.visible_logical_end
                ):
                    group.append(candidate)
                    idx += 1
                    continue
                break

            local_start = group[0].local_start
            local_end = group[-1].local_end
            token_count = local_end - local_start
            block_slices: list[tuple[int, int, int]] = []
            current_block_cursor = 0
            for piece in group:
                piece_start = piece.local_start - local_start
                piece_end = piece.local_end - local_start
                if (
                    block_slices
                    and block_slices[-1][0] == piece.block_id
                    and block_slices[-1][2] == piece_start
                ):
                    block_id, old_start, _ = block_slices[-1]
                    block_slices[-1] = (block_id, old_start, piece_end)
                else:
                    current_block_cursor = piece_start
                    block_slices.append((piece.block_id, current_block_cursor, piece_end))

            slice_ref = ExecutionSliceRef(
                slab_id=slab.slab_id,
                profile=slab.profile,
                token_count=token_count,
                logical_span=(group[0].logical_start, group[-1].logical_end),
                visible_span=(
                    group[0].visible_logical_start,
                    group[-1].visible_logical_end,
                ),
                resident_slice=(resident_cursor, resident_cursor + token_count),
                block_slices=tuple(block_slices),
                local_slice=(local_start, local_end),
                keys_view=slab.keys[..., local_start:local_end, :],
                values_view=slab.values[..., local_start:local_end, :],
                execution_mode=self._handles[group[0].block_id].execution_mode,
            )
            slices.append(slice_ref)
            resident_cursor += token_count
            slab_token_counts[slab.slab_id] = slab_token_counts.get(slab.slab_id, 0) + token_count

        return ResidentStateView(
            total_tokens=resident_cursor,
            slices=tuple(slices),
            topology_epoch=topology_epoch,
            tail_epoch=tail_epoch,
            n_execution_slabs=len(slab_token_counts),
            slab_token_counts=tuple(slab_token_counts.values()),
            fabric_compactions_total=self._compactions_total,
            execution_view_topology_rebuilds_total=execution_view_topology_rebuilds_total,
        )

    def _visible_pieces(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
        *,
        visible_start: int,
    ) -> list[_VisiblePiece]:
        pieces: list[_VisiblePiece] = []
        for handle in ordered_handles:
            logical_cursor = handle.logical_span[0]
            for fragment in handle.fragments:
                slab = self._slabs[fragment.slab_id]
                logical_end = logical_cursor + fragment.token_count
                visible_logical_start = max(logical_cursor, visible_start)
                if visible_logical_start < logical_end:
                    local_start = fragment.local_start + (visible_logical_start - logical_cursor)
                    pieces.append(
                        _VisiblePiece(
                            block_id=handle.block_id,
                            slab_id=fragment.slab_id,
                            local_start=local_start,
                            local_end=fragment.local_end,
                            logical_start=logical_cursor,
                            logical_end=logical_end,
                            visible_logical_start=visible_logical_start,
                            visible_logical_end=logical_end,
                        )
                    )
                logical_cursor = logical_end
                if slab.slab_id != fragment.slab_id:
                    raise RuntimeError("Resident fragment slab lookup mismatch")
        return pieces

    def _remove_handle_fragments(self, handle: ResidentBlockHandle) -> None:
        for fragment in handle.fragments:
            slab = self._slabs.get(fragment.slab_id)
            if slab is None:
                continue
            slab.block_ranges = [
                r
                for r in slab.block_ranges
                if not (
                    r.block_id == handle.block_id
                    and r.local_start == fragment.local_start
                    and r.local_end == fragment.local_end
                )
            ]
            slab.tombstoned_ranges.append((fragment.local_start, fragment.local_end))
            if not slab.block_ranges:
                self._slabs.pop(slab.slab_id, None)
                continue
            slab.logical_span = (
                slab.block_ranges[0].logical_start,
                slab.block_ranges[-1].logical_end,
            )
            self.compact_slab_if_needed(slab.slab_id)
        handle.fragments = ()
        handle.live_bytes = 0

    def compact_slab_if_needed(self, slab_id: int) -> None:
        slab = self._slabs.get(slab_id)
        if slab is None:
            return
        hole_count = 0
        prev_end = None
        for block_range in slab.block_ranges:
            if prev_end is not None and block_range.local_start > prev_end:
                hole_count += 1
            prev_end = block_range.local_end
        slack = slab.used_tokens - slab.live_tokens
        if slack <= 0 and hole_count == 0:
            return
        if slack <= int(0.25 * slab.capacity_tokens) and hole_count <= 2:
            return
        entries: list[tuple[int, mx.array, mx.array, int, int]] = []
        for block_range in slab.block_ranges:
            entries.append(
                (
                    block_range.block_id,
                    slab.keys[..., block_range.local_start : block_range.local_end, :],
                    slab.values[..., block_range.local_start : block_range.local_end, :],
                    block_range.logical_start,
                    block_range.logical_end,
                )
            )
        self._rebuild_slab(slab, entries, capacity_tokens=slab.capacity_tokens)
        self._compactions_total += 1

    def _find_tail_slab(
        self,
        *,
        profile: ResidentProfile,
        dtype: mx.Dtype,
        keys: mx.array,
        values: mx.array,
        logical_span: tuple[int, int],
        capacity: int,
    ) -> ExecutionSlab | None:
        if not self._slabs:
            return None
        tail = max(self._slabs.values(), key=lambda slab: slab.logical_span[1])
        if not _slab_compatible(
            tail,
            profile=profile,
            dtype=dtype,
            keys=keys,
            values=values,
        ):
            return None
        if tail.logical_span[1] != logical_span[0]:
            return None
        if tail.used_tokens + keys.shape[2] > max(tail.capacity_tokens, capacity):
            return None
        return tail

    def _find_insert_slab(
        self,
        *,
        profile: ResidentProfile,
        dtype: mx.Dtype,
        keys: mx.array,
        values: mx.array,
        logical_span: tuple[int, int],
        capacity: int,
    ) -> ExecutionSlab | None:
        candidates = sorted(self._slabs.values(), key=lambda slab: slab.logical_span[0])
        for slab in candidates:
            if not _slab_compatible(
                slab,
                profile=profile,
                dtype=dtype,
                keys=keys,
                values=values,
            ):
                continue
            self.compact_slab_if_needed(slab.slab_id)
            slab = self._slabs.get(slab.slab_id)
            if slab is None:
                continue
            if slab.live_tokens + keys.shape[2] > max(slab.capacity_tokens, capacity):
                continue
            if self._can_insert_logically(slab, logical_span):
                return slab
        return None

    def _can_insert_logically(
        self,
        slab: ExecutionSlab,
        logical_span: tuple[int, int],
    ) -> bool:
        if not slab.block_ranges:
            return True
        for idx, block_range in enumerate(slab.block_ranges):
            if logical_span[1] <= block_range.logical_start:
                if idx == 0:
                    return True
                prev = slab.block_ranges[idx - 1]
                return prev.logical_end <= logical_span[0]
        return slab.block_ranges[-1].logical_end <= logical_span[0]

    def _append_new_block_to_slab(
        self,
        slab: ExecutionSlab,
        *,
        handle: ResidentBlockHandle,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
    ) -> None:
        local_start = slab.used_tokens
        slab.keys = _concat_token_axis([slab.keys, keys.astype(handle.dtype)])
        slab.values = _concat_token_axis([slab.values, values.astype(handle.dtype)])
        slab.used_tokens += keys.shape[2]
        block_range = SlabBlockRange(
            block_id=handle.block_id,
            local_start=local_start,
            local_end=slab.used_tokens,
            logical_start=logical_span[0],
            logical_end=logical_span[1],
        )
        slab.block_ranges.append(block_range)
        slab.logical_span = (
            slab.block_ranges[0].logical_start,
            slab.block_ranges[-1].logical_end,
        )
        handle.fragments = (
            *handle.fragments,
            ExecutionFragment(slab.slab_id, local_start, slab.used_tokens),
        )
        handle.logical_span = (
            (handle.logical_span[0], logical_span[1])
            if handle.fragments[:-1]
            else logical_span
        )
        handle.live_bytes = handle.token_count * handle.per_token_live_bytes

    def _insert_block_into_existing_slab(
        self,
        slab: ExecutionSlab,
        *,
        handle: ResidentBlockHandle,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
    ) -> None:
        insert_idx = 0
        while (
            insert_idx < len(slab.block_ranges)
            and slab.block_ranges[insert_idx].logical_start < logical_span[0]
        ):
            insert_idx += 1

        entries: list[tuple[int, mx.array, mx.array, int, int]] = []
        inserted = False
        for idx, block_range in enumerate(slab.block_ranges):
            if idx == insert_idx:
                entries.append(
                    (
                        handle.block_id,
                        keys.astype(handle.dtype),
                        values.astype(handle.dtype),
                        logical_span[0],
                        logical_span[1],
                    )
                )
                inserted = True
            entries.append(
                (
                    block_range.block_id,
                    slab.keys[..., block_range.local_start : block_range.local_end, :],
                    slab.values[..., block_range.local_start : block_range.local_end, :],
                    block_range.logical_start,
                    block_range.logical_end,
                )
            )
        if not inserted:
            entries.append(
                (
                    handle.block_id,
                    keys.astype(handle.dtype),
                    values.astype(handle.dtype),
                    logical_span[0],
                    logical_span[1],
                )
            )
        self._rebuild_slab(
            slab,
            entries,
            capacity_tokens=max(slab.capacity_tokens, slab.live_tokens + keys.shape[2]),
        )

    def _rebuild_slab(
        self,
        slab: ExecutionSlab,
        entries: list[tuple[int, mx.array, mx.array, int, int]],
        *,
        capacity_tokens: int,
    ) -> None:
        key_parts = [entry[1] for entry in entries]
        value_parts = [entry[2] for entry in entries]
        slab.keys = _concat_token_axis(key_parts)
        slab.values = _concat_token_axis(value_parts)
        slab.capacity_tokens = max(capacity_tokens, slab.keys.shape[2])
        slab.used_tokens = slab.keys.shape[2]
        slab.tombstoned_ranges = []
        slab.block_ranges = []
        cursor = 0
        for block_id, keys, _values, logical_start, logical_end in entries:
            next_cursor = cursor + keys.shape[2]
            slab.block_ranges.append(
                SlabBlockRange(
                    block_id=block_id,
                    local_start=cursor,
                    local_end=next_cursor,
                    logical_start=logical_start,
                    logical_end=logical_end,
                )
            )
            cursor = next_cursor
        slab.logical_span = (
            slab.block_ranges[0].logical_start,
            slab.block_ranges[-1].logical_end,
        )
        self._rebind_slab_fragments(slab)

    def _rebind_slab_fragments(self, slab: ExecutionSlab) -> None:
        for block_range in slab.block_ranges:
            handle = self._handles[block_range.block_id]
            fragment = ExecutionFragment(
                slab.slab_id,
                block_range.local_start,
                block_range.local_end,
            )
            replaced = False
            fragments = list(handle.fragments)
            for idx, existing in enumerate(fragments):
                if existing.slab_id == slab.slab_id:
                    fragments[idx] = fragment
                    replaced = True
                    break
            if not replaced:
                insert_at = 0
                while (
                    insert_at < len(fragments)
                    and self._slabs[fragments[insert_at].slab_id].logical_span[0]
                    < block_range.logical_start
                ):
                    insert_at += 1
                fragments.insert(insert_at, fragment)
            handle.fragments = tuple(fragments)
            handle.live_bytes = handle.token_count * handle.per_token_live_bytes

    def _new_slab(
        self,
        *,
        profile: ResidentProfile,
        dtype: mx.Dtype,
        capacity: int,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
    ) -> ExecutionSlab:
        slab = ExecutionSlab(
            slab_id=self._next_slab_id,
            profile=profile,
            logical_span=logical_span,
            keys=keys.astype(dtype),
            values=values.astype(dtype),
            capacity_tokens=max(capacity, keys.shape[2]),
            used_tokens=keys.shape[2],
            dtype=dtype,
        )
        self._slabs[slab.slab_id] = slab
        self._next_slab_id += 1
        return slab


class TurboQuantResidentBackend:
    """TurboQuant resident backend backed by a persistent exact execution fabric."""

    def __init__(
        self,
        *,
        safe_bits: int,
        aggr_bits: int,
        safe_execution_mode: ResidentExecutionMode = ResidentExecutionMode.DEQUANTIZE_ON_READ,
        aggr_execution_mode: ResidentExecutionMode = ResidentExecutionMode.DEQUANTIZE_ON_READ,
    ) -> None:
        del safe_bits, aggr_bits
        self._safe_execution_mode = safe_execution_mode
        self._aggr_execution_mode = aggr_execution_mode
        self.fabric = ResidentExecutionFabric()

    @property
    def compactions_total(self) -> int:
        return self.fabric.compactions_total

    def reset(self) -> None:
        self.fabric.reset()

    def descriptor(self, profile: ResidentProfile) -> ResidentProfileDescriptor:
        if profile is ResidentProfile.TQ_SAFE:
            return ResidentProfileDescriptor(
                profile=profile,
                slab_capacity_tokens=256,
                execution_mode=self._safe_execution_mode,
                fidelity_rank=2,
                memory_cost_class=2,
            )
        if profile is ResidentProfile.TQ_AGGR:
            return ResidentProfileDescriptor(
                profile=profile,
                slab_capacity_tokens=128,
                execution_mode=self._aggr_execution_mode,
                fidelity_rank=1,
                memory_cost_class=1,
            )
        raise ValueError("Resident backend descriptors are not defined for evicted blocks")

    def create_handle(
        self,
        *,
        block_id: int,
        profile: ResidentProfile,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
    ) -> ResidentBlockHandle:
        descriptor = self.descriptor(profile)
        handle = ResidentBlockHandle(
            block_id=block_id,
            profile=profile,
            logical_span=logical_span,
            fragments=(),
            dtype=keys.dtype,
            execution_mode=descriptor.execution_mode,
            slab_capacity_tokens=descriptor.slab_capacity_tokens,
            per_token_live_bytes=_per_token_live_bytes(keys, values),
            live_bytes=0,
        )
        self.fabric.register_handle(handle)
        self.fabric.allocate_block(
            handle,
            logical_span=logical_span,
            keys=keys,
            values=values,
            allow_tail_fast=True,
        )
        handle.live_bytes = handle.token_count * handle.per_token_live_bytes
        return handle

    def append_tokens(
        self,
        handle: ResidentBlockHandle,
        *,
        logical_span: tuple[int, int],
        keys: mx.array,
        values: mx.array,
        dtype: mx.Dtype,
    ) -> ResidentBlockHandle:
        del dtype
        self.fabric.append_to_tail(
            handle,
            logical_span=logical_span,
            keys=keys.astype(handle.dtype),
            values=values.astype(handle.dtype),
        )
        handle.logical_span = logical_span
        handle.live_bytes = handle.token_count * handle.per_token_live_bytes
        return handle

    def convert_profile(
        self,
        handle: ResidentBlockHandle,
        *,
        profile: ResidentProfile,
        dtype: mx.Dtype,
    ) -> ResidentBlockHandle:
        if handle.profile is profile:
            return handle
        keys, values = self.materialize(handle)
        descriptor = self.descriptor(profile)
        self.fabric.move_block(
            handle,
            profile=profile,
            logical_span=handle.logical_span,
            keys=keys.astype(dtype),
            values=values.astype(dtype),
            execution_mode=descriptor.execution_mode,
            slab_capacity_tokens=descriptor.slab_capacity_tokens,
        )
        return handle

    def evict_handle(self, handle: ResidentBlockHandle) -> None:
        self.fabric.evict_block(handle)

    def materialize(self, handle: ResidentBlockHandle) -> tuple[mx.array, mx.array]:
        return self.fabric.materialize(handle)

    def debug_materialize(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
    ) -> tuple[mx.array | None, mx.array | None]:
        return self.fabric.debug_materialize(ordered_handles)

    def query_full_view(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
        *,
        topology_epoch: int,
        tail_epoch: int,
        execution_view_topology_rebuilds_total: int,
    ) -> Any:
        return self.fabric.query_full_view(
            ordered_handles,
            topology_epoch=topology_epoch,
            tail_epoch=tail_epoch,
            execution_view_topology_rebuilds_total=execution_view_topology_rebuilds_total,
        )

    def query_window_view(
        self,
        ordered_handles: tuple[ResidentBlockHandle, ...],
        *,
        logical_offset: int,
        query_tokens: int,
        window_size: int,
        topology_epoch: int,
        tail_epoch: int,
        execution_view_topology_rebuilds_total: int,
    ) -> Any:
        return self.fabric.query_window_view(
            ordered_handles,
            logical_offset=logical_offset,
            query_tokens=query_tokens,
            window_size=window_size,
            topology_epoch=topology_epoch,
            tail_epoch=tail_epoch,
            execution_view_topology_rebuilds_total=execution_view_topology_rebuilds_total,
        )


__all__ = [
    "ExecutionFragment",
    "ExecutionSlab",
    "ResidentBackend",
    "ResidentBlockHandle",
    "ResidentExecutionFabric",
    "ResidentExecutionMode",
    "ResidentProfileDescriptor",
    "SlabBlockRange",
    "TurboQuantResidentBackend",
]
