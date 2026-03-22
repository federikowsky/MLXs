"""Ghost metadata store for evicted adaptive blocks."""

from __future__ import annotations

from dataclasses import replace

from mlxs.adaptive_kv.block_types import BlockRecord, GhostRecord


class AdaptiveGhostStore:
    """Stores lightweight metadata for evicted blocks."""

    def __init__(self) -> None:
        self._ghosts: dict[int, GhostRecord] = {}

    def create(self, block: BlockRecord, *, step: int) -> GhostRecord:
        ghost = self._ghosts.get(block.block_id)
        if ghost is None:
            ghost = GhostRecord(
                block_id=block.block_id,
                source_start=block.source_start,
                source_end=block.source_end,
                segment_id=block.segment_id,
                last_evicted_step=step,
                last_score=block.score.composite,
            )
        else:
            ghost = replace(
                ghost,
                last_evicted_step=step,
                evict_count=ghost.evict_count + 1,
                last_score=block.score.composite,
                recently_reactivated=False,
            )
        self._ghosts[block.block_id] = ghost
        return ghost

    def mark_reactivated(self, block_id: int) -> None:
        ghost = self._ghosts.get(block_id)
        if ghost is None:
            return
        self._ghosts[block_id] = replace(ghost, recently_reactivated=True)

    def get(self, block_id: int) -> GhostRecord | None:
        return self._ghosts.get(block_id)

    def has(self, block_id: int) -> bool:
        return block_id in self._ghosts

    def snapshot(self) -> dict[int, GhostRecord]:
        return dict(self._ghosts)

