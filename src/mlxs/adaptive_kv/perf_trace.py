"""Lightweight gated performance attribution for Adaptive KV hot paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _PerfStat:
    total_ns: int = 0
    calls: int = 0
    count: int = 0


@dataclass(slots=True)
class AdaptiveKVPerfTrace:
    enabled: bool = False
    sync_enabled: bool = False
    _stats: dict[str, _PerfStat] = field(default_factory=dict)

    def record_ns(self, name: str, elapsed_ns: int) -> None:
        if not self.enabled:
            return
        stat = self._stats.setdefault(name, _PerfStat())
        stat.total_ns += int(elapsed_ns)
        stat.calls += 1

    def increment(self, name: str, delta: int = 1) -> None:
        if not self.enabled:
            return
        stat = self._stats.setdefault(name, _PerfStat())
        stat.count += int(delta)

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        totals = {name: stat.total_ns for name, stat in self._stats.items() if stat.total_ns > 0}
        traced_total_ns = sum(totals.values())
        stats: dict[str, Any] = {}
        for name in sorted(self._stats):
            stat = self._stats[name]
            row: dict[str, Any] = {}
            if stat.total_ns > 0:
                row["total_ns"] = stat.total_ns
                row["total_ms"] = round(stat.total_ns / 1_000_000.0, 4)
                row["calls"] = stat.calls
                row["avg_ns"] = stat.total_ns / max(1, stat.calls)
                row["avg_ms"] = round(row["avg_ns"] / 1_000_000.0, 6)
                row["share_of_traced_ns"] = (
                    stat.total_ns / traced_total_ns if traced_total_ns > 0 else 0.0
                )
            if stat.count > 0:
                row["count"] = stat.count
            stats[name] = row
        return {
            "enabled": True,
            "sync_enabled": self.sync_enabled,
            "traced_total_ns": traced_total_ns,
            "traced_total_ms": round(traced_total_ns / 1_000_000.0, 4),
            "stats": stats,
        }
