"""Layer 4 observability exposure surfaces."""

from __future__ import annotations

import logging
from typing import Any

from mlxs.observability.logger import setup_logging
from mlxs.observability.metrics import InMemoryMetrics, create_metrics

logger = logging.getLogger(__name__)


def configure_observability(config: Any) -> Any:
    """Configure Layer 4 logging and metrics exposure."""
    setup_logging(
        level=config.observability.log_level,
        json_format=False,
    )
    metrics = create_metrics(config.observability)
    if config.observability.metrics_enabled:
        logger.info(
            "Metrics enabled on main app route /metrics; metrics_port=%s is compatibility-only in Phase 5.",
            config.observability.metrics_port,
        )
    return metrics


def record_counter(runtime: Any, name: str, value: float = 1.0, **labels: str) -> None:
    metrics = getattr(runtime, "metrics", None)
    counter = getattr(metrics, "counter", None)
    if callable(counter):
        counter(name, value, **labels)


def metrics_snapshot(runtime: Any) -> dict[str, Any]:
    config = getattr(runtime, "config", None)
    observability = getattr(config, "observability", None)
    enabled = bool(observability and observability.metrics_enabled)
    snapshot: dict[str, Any] = {
        "enabled": enabled,
        "route_mode": "main_app",
        "metrics_port_compatibility_only": True,
        "metrics_port": getattr(observability, "metrics_port", None),
    }

    metrics = getattr(runtime, "metrics", None)
    if isinstance(metrics, InMemoryMetrics):
        snapshot["backend"] = "in_memory"
        snapshot["counters"] = dict(metrics._counters)
        snapshot["gauges"] = dict(metrics._gauges)
        snapshot["histograms"] = {key: list(values) for key, values in metrics._histograms.items()}
    else:
        snapshot["backend"] = "noop"
    return snapshot


async def metrics_endpoint(request: Any) -> Any:
    from starlette.responses import JSONResponse

    runtime = getattr(request.app.state, "runtime", getattr(request.app.state, "deps", None))
    return JSONResponse(metrics_snapshot(runtime))
