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
    snapshot["prompt_cache"] = _prompt_cache_snapshot(runtime)
    return snapshot


def _prompt_cache_snapshot(runtime: Any) -> dict[str, Any]:
    config = getattr(runtime, "config", None)
    prompt_cache_config = getattr(config, "prompt_cache", None)
    payload: dict[str, Any] = {
        "enabled": bool(prompt_cache_config and prompt_cache_config.enabled),
        "max_entries": getattr(prompt_cache_config, "max_entries", None),
        "max_bytes": getattr(prompt_cache_config, "max_bytes", None),
        "trim_on_rss_gb": getattr(prompt_cache_config, "trim_on_rss_gb", None),
        "target_rss_ratio": getattr(prompt_cache_config, "target_rss_ratio", None),
        "on_memory_ceiling": (
            getattr(getattr(prompt_cache_config, "on_memory_ceiling", None), "value", None)
        ),
    }

    prompt_cache = getattr(runtime, "prompt_cache", None)
    stats_fn = getattr(prompt_cache, "stats", None)
    if not callable(stats_fn):
        payload["available"] = False
        return payload

    stats = stats_fn()
    lookups = stats.hit_count + stats.miss_count
    payload.update(
        {
            "available": True,
            "hit_count": stats.hit_count,
            "miss_count": stats.miss_count,
            "lookup_count": lookups,
            "hit_ratio": (stats.hit_count / lookups) if lookups > 0 else None,
            "eviction_count": stats.eviction_count,
            "entry_count": stats.entry_count,
            "total_bytes": stats.total_bytes,
        }
    )
    return payload


async def metrics_endpoint(request: Any) -> Any:
    from starlette.responses import JSONResponse

    runtime = getattr(request.app.state, "runtime", getattr(request.app.state, "deps", None))
    return JSONResponse(metrics_snapshot(runtime))
