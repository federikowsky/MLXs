"""Metrics implementations — no-op and real (NFR2, §5.3, AC16).

The no-op backend is the default. When metrics are enabled, ``create_metrics``
returns a real implementation that collects counters, gauges, and histograms.

Both satisfy ``MetricsProtocol``. The no-op backend has zero overhead
in the hot path — methods are empty, no dict lookups, no locks.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

from mlxs.config.schema import ObservabilityConfig


class NoOpMetrics:
    """Metrics sink that discards everything. Zero hot-path overhead (O2).

    Satisfies ``MetricsProtocol``.
    """

    __slots__ = ()

    def counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        pass

    def gauge(self, name: str, value: float, **labels: str) -> None:
        pass

    def histogram(self, name: str, value: float, **labels: str) -> None:
        pass


@dataclass
class InMemoryMetrics:
    """Simple in-memory metrics collector for development and testing.

    Thread-safe. Satisfies ``MetricsProtocol``.

    Not intended for production — a real metrics backend (e.g. Prometheus)
    would replace this in Phase 5 (server).
    """

    _counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _gauges: dict[str, float] = field(default_factory=dict)
    _histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = _make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        key = _make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def histogram(self, name: str, value: float, **labels: str) -> None:
        key = _make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    def get_counter(self, name: str, **labels: str) -> float:
        """Read a counter value (for testing)."""
        return self._counters.get(_make_key(name, labels), 0.0)

    def get_gauge(self, name: str, **labels: str) -> float:
        """Read a gauge value (for testing)."""
        return self._gauges.get(_make_key(name, labels), 0.0)

    def get_histogram(self, name: str, **labels: str) -> list[float]:
        """Read histogram observations (for testing)."""
        return list(self._histograms.get(_make_key(name, labels), []))


def _make_key(name: str, labels: dict[str, str]) -> str:
    """Build a metric key from name and labels."""
    if not labels:
        return name
    suffix = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{suffix}}}"


def create_metrics(config: ObservabilityConfig) -> NoOpMetrics | InMemoryMetrics:
    """Factory: return the appropriate metrics backend based on config.

    Args:
        config: Observability configuration section.

    Returns:
        ``InMemoryMetrics`` when enabled, ``NoOpMetrics`` otherwise.
    """
    if config.metrics_enabled:
        return InMemoryMetrics()
    return NoOpMetrics()
