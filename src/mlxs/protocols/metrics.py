"""Metrics protocol — contract for observability instrumentation (NFR2, §5.3).

Modules emit metrics through this protocol. The observability module provides
a no-op implementation (default) and a real implementation (when enabled).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsProtocol(Protocol):
    """Structural contract for metrics emission.

    Supports counters, gauges, and histograms. All methods are no-ops when
    metrics are disabled, avoiding any hot-path overhead (O2).
    """

    def counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Increment a counter.

        Args:
            name: Metric name (e.g. ``prompt_cache_hit_count``).
            value: Amount to increment (default 1).
            **labels: Optional key-value labels.
        """
        ...

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge to a value.

        Args:
            name: Metric name (e.g. ``batch_size_decode``).
            value: Current value.
            **labels: Optional key-value labels.
        """
        ...

    def histogram(self, name: str, value: float, **labels: str) -> None:
        """Record a value in a histogram.

        Args:
            name: Metric name (e.g. ``time_to_first_token_seconds``).
            value: Observed value.
            **labels: Optional key-value labels.
        """
        ...
