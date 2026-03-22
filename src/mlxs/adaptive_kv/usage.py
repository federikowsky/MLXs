"""Windowed adaptive usage collection."""

from __future__ import annotations

import time

import mlx.core as mx


class AdaptiveUsageCollector:
    """Collects per-window usage signals keyed by block id."""

    def __init__(self) -> None:
        self._window_usage: dict[int, float] = {}
        self._window_batches: list[tuple[tuple[int, ...], mx.array]] = []

    def record(self, block_id: int, value: float) -> None:
        if value <= 0:
            return
        self._window_usage[block_id] = self._window_usage.get(block_id, 0.0) + value

    def record_batch(self, block_ids: tuple[int, ...], values: mx.array) -> None:
        if not block_ids or values.size == 0:
            return
        self._window_batches.append((block_ids, values))

    def snapshot_and_reset(
        self,
        *,
        timing_acc: dict[str, int] | None = None,
    ) -> dict[int, float]:
        usage = dict(self._window_usage)
        if self._window_batches:
            block_ids: list[int] = []
            value_batches: list[mx.array] = []
            for batch_block_ids, batch_values in self._window_batches:
                block_ids.extend(batch_block_ids)
                value_batches.append(batch_values)
            usage_values = (
                value_batches[0]
                if len(value_batches) == 1
                else mx.concatenate(value_batches, axis=0)
            )
            if timing_acc is not None:
                t0 = time.perf_counter_ns()
                mx.eval(usage_values)
                timing_acc["eval_ns"] = timing_acc.get("eval_ns", 0) + (
                    time.perf_counter_ns() - t0
                )
                t_host = time.perf_counter_ns()
                host_values = usage_values.tolist()
                timing_acc["host_ns"] = timing_acc.get("host_ns", 0) + (
                    time.perf_counter_ns() - t_host
                )
            else:
                mx.eval(usage_values)
                host_values = usage_values.tolist()
            for block_id, value in zip(block_ids, host_values, strict=True):
                if value <= 0:
                    continue
                usage[block_id] = usage.get(block_id, 0.0) + float(value)

        self._window_usage = {}
        self._window_batches = []
        if not usage:
            return {}

        max_usage = max(usage.values()) or 1.0
        normalized = {
            block_id: min(1.0, usage / max_usage)
            for block_id, usage in usage.items()
        }
        return normalized
