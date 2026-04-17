"""Layer 4 observability exposure surfaces."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any

import mlx.core as mx

from mlxs.observability.logger import setup_logging
from mlxs.observability.metrics import InMemoryMetrics, create_metrics

logger = logging.getLogger(__name__)
_UNSET = object()
_PROCESS_RSS_CACHE_TTL_S = 0.25
_PROCESS_RSS_CACHE: dict[int, tuple[float, int]] = {}
_MAX_RECOMMENDED_WORKING_SET_SIZE_BYTES: int | None | object = _UNSET


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
    process_snapshot = _process_snapshot()
    snapshot["process"] = process_snapshot
    snapshot["startup"] = _startup_snapshot(runtime)
    snapshot["runtime"] = _runtime_snapshot(runtime)
    prompt_cache_snapshot = _prompt_cache_snapshot(runtime)
    snapshot["prompt_cache"] = prompt_cache_snapshot
    snapshot["memory"] = _memory_snapshot(
        runtime,
        process_snapshot=process_snapshot,
        prompt_cache_snapshot=prompt_cache_snapshot,
    )
    snapshot["request_queue"] = _request_queue_snapshot(runtime)
    snapshot["request_outcomes"] = _request_outcomes_snapshot(snapshot)
    return snapshot


def _process_snapshot() -> dict[str, Any]:
    now = time.time()
    payload: dict[str, Any] = {
        "available": True,
        "pid": os.getpid(),
        "rss_kb": None,
        "rss_bytes": None,
        "rss_source": "ps",
        "rss_sample_age_s": None,
    }
    try:
        pid = int(payload["pid"])
        cached = _PROCESS_RSS_CACHE.get(pid)
        if cached is not None and (now - cached[0]) <= _PROCESS_RSS_CACHE_TTL_S:
            rss_kb = cached[1]
            payload["rss_sample_age_s"] = now - cached[0]
        else:
            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)],
                text=True,
            ).strip()
            rss_kb = int(out)
            _PROCESS_RSS_CACHE[pid] = (now, rss_kb)
            payload["rss_sample_age_s"] = 0.0
        payload["rss_kb"] = rss_kb
        payload["rss_bytes"] = rss_kb * 1024
    except Exception:
        payload["available"] = False
    return payload


def _memory_snapshot(
    runtime: Any,
    *,
    process_snapshot: dict[str, Any],
    prompt_cache_snapshot: dict[str, Any],
) -> dict[str, Any]:
    config = getattr(runtime, "config", None)
    memory = getattr(config, "memory", None)
    prompt_cache = getattr(config, "prompt_cache", None)
    payload: dict[str, Any] = {
        "available": True,
        "wired_limit_configured_bytes": getattr(memory, "wired_limit", None),
        "max_recommended_working_set_size_bytes": None,
        "effective_wired_limit_bytes": None,
        "rss_bytes": process_snapshot.get("rss_bytes"),
        "rss_to_effective_wired_limit_ratio": None,
        "prompt_cache_total_bytes": prompt_cache_snapshot.get("total_bytes"),
        "prompt_cache_to_rss_ratio": None,
        "prompt_cache_trim_on_rss_gb": getattr(prompt_cache, "trim_on_rss_gb", None),
        "prompt_cache_target_rss_ratio": getattr(prompt_cache, "target_rss_ratio", None),
        "prompt_cache_on_memory_ceiling": (
            getattr(getattr(prompt_cache, "on_memory_ceiling", None), "value", None)
        ),
    }
    global _MAX_RECOMMENDED_WORKING_SET_SIZE_BYTES
    try:
        if _MAX_RECOMMENDED_WORKING_SET_SIZE_BYTES is _UNSET:
            info = mx.device_info()
            _MAX_RECOMMENDED_WORKING_SET_SIZE_BYTES = info.get(
                "max_recommended_working_set_size"
            )
        if _MAX_RECOMMENDED_WORKING_SET_SIZE_BYTES is not _UNSET:
            payload["max_recommended_working_set_size_bytes"] = _MAX_RECOMMENDED_WORKING_SET_SIZE_BYTES
    except Exception:
        payload["available"] = False

    effective_wired_limit = (
        payload["wired_limit_configured_bytes"] or payload["max_recommended_working_set_size_bytes"]
    )
    payload["effective_wired_limit_bytes"] = effective_wired_limit

    rss_bytes = payload["rss_bytes"]
    if isinstance(rss_bytes, int) and isinstance(effective_wired_limit, int) and effective_wired_limit > 0:
        payload["rss_to_effective_wired_limit_ratio"] = rss_bytes / effective_wired_limit
    prompt_cache_total_bytes = payload["prompt_cache_total_bytes"]
    if isinstance(rss_bytes, int) and rss_bytes > 0 and isinstance(prompt_cache_total_bytes, int):
        payload["prompt_cache_to_rss_ratio"] = prompt_cache_total_bytes / rss_bytes
    return payload


def _startup_snapshot(runtime: Any) -> dict[str, Any]:
    lifecycle = getattr(runtime, "lifecycle", None)
    config = getattr(runtime, "config", None)
    model = getattr(config, "model", None)
    generate = getattr(config, "generate", None)
    payload: dict[str, Any] = {
        "available": lifecycle is not None,
        "lazy_load_configured": getattr(model, "lazy_load", None),
        "preload_configured": getattr(model, "preload", None),
        "warmup_after_load": getattr(generate, "warmup_after_load", None),
        # Layer 4 bootstrap always materializes model residency before ready.
        "effective_eager_residency": True,
    }
    if lifecycle is None:
        return payload
    payload.update(
        {
            "instance_id": getattr(lifecycle, "instance_id", None),
            "model_id": getattr(lifecycle, "model_id", None),
            "ready": getattr(lifecycle, "ready", None),
            "stopped": getattr(lifecycle, "stopped", None),
            "last_error": getattr(lifecycle, "last_error", None),
            "started_at": getattr(lifecycle, "started_at", None),
            "ready_at": getattr(lifecycle, "ready_at", None),
            "stopped_at": getattr(lifecycle, "stopped_at", None),
        }
    )
    started_at = payload["started_at"]
    ready_at = payload["ready_at"]
    if isinstance(started_at, (int, float)):
        payload["uptime_s"] = max(0.0, time.time() - float(started_at))
    else:
        payload["uptime_s"] = None
    if isinstance(started_at, (int, float)) and isinstance(ready_at, (int, float)):
        payload["startup_duration_s"] = max(0.0, float(ready_at) - float(started_at))
    else:
        payload["startup_duration_s"] = None
    return payload


def _runtime_snapshot(runtime: Any) -> dict[str, Any]:
    config = getattr(runtime, "config", None)
    model = getattr(config, "model", None)
    generate = getattr(config, "generate", None)
    batch = getattr(config, "batch", None)
    server = getattr(config, "server", None)
    return {
        "available": config is not None,
        "model_path": getattr(model, "model_path", None),
        "model_type": getattr(model, "model_type", None),
        "batch_host_enabled": getattr(getattr(runtime, "batch_host", None), "enabled", False),
        "prefill_batch_size": getattr(batch, "prefill_batch_size", None),
        "completion_batch_size": getattr(batch, "completion_batch_size", None),
        "prefill_step_size": getattr(batch, "prefill_step_size", None),
        "max_concurrent_requests": getattr(server, "max_concurrent_requests", None),
        "max_queue_size": getattr(server, "max_queue_size", None),
        "request_timeout_seconds": getattr(server, "request_timeout", None),
        "compile_decode": getattr(generate, "compile_decode", None),
        "warmup_after_load": getattr(generate, "warmup_after_load", None),
        "stream_policy": getattr(getattr(generate, "stream_policy", None), "value", None),
        "clear_cache_interval": getattr(generate, "clear_cache_interval", None),
    }


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


def _request_queue_snapshot(runtime: Any) -> dict[str, Any]:
    config = getattr(runtime, "config", None)
    server = getattr(config, "server", None)
    queue = getattr(runtime, "request_queue", None)
    payload: dict[str, Any] = {
        "configured_max_concurrent_requests": getattr(server, "max_concurrent_requests", None),
        "configured_max_queue_size": getattr(server, "max_queue_size", None),
        "configured_request_timeout_seconds": getattr(server, "request_timeout", None),
        "batch_host_enabled": getattr(getattr(runtime, "batch_host", None), "enabled", False),
    }
    if queue is None:
        payload["available"] = False
        return payload

    active_count = getattr(queue, "active_count", None)
    pending_count = getattr(queue, "pending_count", None)
    payload.update(
        {
            "available": True,
            "active_count": active_count,
            "pending_count": pending_count,
            "peak_active_count": getattr(queue, "peak_active_count", None),
            "peak_pending_count": getattr(queue, "peak_pending_count", None),
            "peak_inflight_count": (
                (getattr(queue, "peak_active_count", 0) or 0)
                + (getattr(queue, "peak_pending_count", 0) or 0)
            ),
            "inflight_count": (
                (active_count or 0) + (pending_count or 0)
                if active_count is not None and pending_count is not None
                else None
            ),
            "is_full": getattr(queue, "is_full", None),
        }
    )
    return payload


def _request_outcomes_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    counters = snapshot.get("counters", {})
    total = counters.get("product_requests_total{surface=http}", 0.0)
    completed = counters.get("product_requests_completed_total{surface=http}", 0.0)
    rejected = counters.get("product_requests_rejected_total{surface=http}", 0.0)
    timed_out = counters.get("product_requests_timeout_total{surface=http}", 0.0)
    failed = counters.get("product_requests_failed_total{surface=http}", 0.0)
    cancelled = counters.get("product_requests_cancelled_total{surface=http}", 0.0)
    return {
        "total": total,
        "completed": completed,
        "rejected": rejected,
        "timed_out": timed_out,
        "failed": failed,
        "cancelled": cancelled,
        "incomplete": max(0.0, total - completed - rejected - timed_out - failed - cancelled),
    }


async def metrics_endpoint(request: Any) -> Any:
    from starlette.responses import JSONResponse

    runtime = getattr(request.app.state, "runtime", getattr(request.app.state, "deps", None))
    return JSONResponse(metrics_snapshot(runtime))
