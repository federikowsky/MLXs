"""Robust order statistics for repeated benchmark trials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median, pstdev


@dataclass(frozen=True, slots=True)
class TrialStats:
    n: int
    mean: float
    median: float
    stdev: float
    min: float
    max: float
    p05: float
    p95: float


def _percentile_nearest_rank(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize_trials(values: list[float]) -> TrialStats:
    """Classic summary: mean, median, sample stdev, min/max, p05/p95."""
    xs = [float(x) for x in values if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        nan = float("nan")
        return TrialStats(0, nan, nan, nan, nan, nan, nan, nan)
    s = sorted(xs)
    sd = pstdev(xs) if n > 1 else 0.0
    return TrialStats(
        n=n,
        mean=mean(xs),
        median=median(xs),
        stdev=sd,
        min=s[0],
        max=s[-1],
        p05=_percentile_nearest_rank(s, 5.0),
        p95=_percentile_nearest_rank(s, 95.0),
    )


def ratio_median(a: TrialStats, b: TrialStats) -> float:
    """a.median / b.median (e.g. MLXs / mlx-lm)."""
    if b.median == 0 or math.isnan(b.median):
        return float("nan")
    return a.median / b.median
