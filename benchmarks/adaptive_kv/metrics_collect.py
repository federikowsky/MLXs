"""Serialize InMemoryMetrics and adaptive debug snapshots for JSON output."""

from __future__ import annotations

import math
import resource
import sys
from typing import Any

import mlx.core as mx


def _rusage_maxrss_bytes() -> int | None:
    """Best-effort peak RSS for this process (platform-dependent)."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def snapshot_in_memory_metrics(m: Any) -> dict[str, Any]:
    """Dump all counters, gauges, and histograms (reads private fields)."""
    with m._lock:
        counters = dict(m._counters)
        gauges = dict(m._gauges)
        histograms = {k: list(v) for k, v in m._histograms.items()}
    return {"counters": counters, "gauges": gauges, "histograms": histograms}


def histogram_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    sorted_v = sorted(values)
    n = len(sorted_v)

    def pct(p: float) -> float:
        idx = int(math.ceil(p * n) - 1)
        idx = max(0, min(n - 1, idx))
        return float(sorted_v[idx])

    return {
        "count": n,
        "sum": float(sum(sorted_v)),
        "mean": float(sum(sorted_v) / n),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": float(sorted_v[-1]),
    }


def sanitize_debug_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Make debug_snapshot JSON-safe (drop non-serializable values)."""
    out = {
        "decode_steps": snap.get("decode_steps"),
        "pressure_state": snap.get("pressure_state"),
        "resident_bytes": snap.get("resident_bytes"),
        "block_count": len(snap.get("blocks", [])),
    }
    ap = snap.get("attention_path")
    if isinstance(ap, dict):
        out["attention_path"] = dict(ap)
    blocks = snap.get("blocks")
    if isinstance(blocks, list):
        tier_counts: dict[str, int] = {}
        for b in blocks:
            if isinstance(b, dict) and "tier" in b:
                t = str(b["tier"])
                tier_counts[t] = tier_counts.get(t, 0) + 1
        out["tier_counts"] = tier_counts
        out["blocks"] = blocks
    ghosts = snap.get("ghosts")
    if isinstance(ghosts, dict):
        out["ghost_count"] = len(ghosts)
        out["ghost_block_ids"] = sorted(int(k) for k in ghosts)
    return out


def estimated_full_kv_bytes_per_prompt_token(model: Any) -> int | None:
    """Rough Llama KV size: layers * 2 * n_kv_heads * head_dim * dtype_bytes."""
    args = getattr(model, "args", None)
    if args is None:
        return None
    n_layers = getattr(args, "num_hidden_layers", None)
    n_kv = getattr(args, "num_key_value_heads", None)
    hidden = getattr(args, "hidden_size", None)
    n_heads = getattr(args, "num_attention_heads", None)
    head_dim = getattr(args, "head_dim", None)
    if n_layers is None or n_kv is None or hidden is None or n_heads is None:
        return None
    if head_dim is None:
        head_dim = hidden // n_heads
    # Match common bf16/fp16 KV (2 bytes); may differ under quantization.
    element_bytes = 2
    per_layer = 2 * n_kv * head_dim * element_bytes
    return int(n_layers * per_layer)


def environment_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "mlx": getattr(mx, "__version__", "unknown"),
        "platform": sys.platform,
    }
