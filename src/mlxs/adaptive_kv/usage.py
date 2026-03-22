"""Windowed adaptive usage collection."""

from __future__ import annotations


class AdaptiveUsageCollector:
    """Collects per-window usage signals keyed by block id."""

    def __init__(self) -> None:
        self._window_usage: dict[int, float] = {}

    def record(self, block_id: int, value: float) -> None:
        if value <= 0:
            return
        self._window_usage[block_id] = self._window_usage.get(block_id, 0.0) + value

    def snapshot_and_reset(self) -> dict[int, float]:
        if not self._window_usage:
            return {}
        max_usage = max(self._window_usage.values()) or 1.0
        normalized = {
            block_id: min(1.0, usage / max_usage)
            for block_id, usage in self._window_usage.items()
        }
        self._window_usage = {}
        return normalized

