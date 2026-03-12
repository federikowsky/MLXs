"""Observability module — structured logging and metrics (NFR2, §5.3)."""

from mlxs.observability.logger import setup_logging
from mlxs.observability.metrics import NoOpMetrics, create_metrics

__all__ = ["NoOpMetrics", "create_metrics", "setup_logging"]
