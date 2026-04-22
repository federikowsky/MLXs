"""Shared helpers for scheduler-level AC2 probes and direct comparisons."""

from __future__ import annotations

import statistics
from typing import Any

SUMMARY_FIELDS = (
    "requests_per_s",
    "generated_tok_per_s",
    "p50_ttft_s",
    "p95_completion_s",
    "rss_bytes",
    "mlx_peak_memory_bytes",
)


def p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def summarize_probe_trials(trials: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in SUMMARY_FIELDS:
        values = [float(trial[field]) for trial in trials]
        summary[field] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }
    return summary


def compare_prompt_results(
    mlxs_results: list[dict[str, Any]],
    mlx_lm_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare prompt-wise scheduler summaries for MLXs and mlx-lm."""
    mlx_lm_by_prompt = {int(result["prompt_target"]): result for result in mlx_lm_results}
    comparisons: list[dict[str, Any]] = []
    for mlxs_result in mlxs_results:
        prompt_target = int(mlxs_result["prompt_target"])
        baseline = mlx_lm_by_prompt[prompt_target]
        ratio_fields = (
            "requests_per_s",
            "generated_tok_per_s",
            "p50_ttft_s",
            "p95_completion_s",
        )
        ratios = {
            f"mlxs_over_mlx_lm_{field}_median_ratio": (
                float(mlxs_result["summary"][field]["median"])
                / float(baseline["summary"][field]["median"])
            )
            for field in ratio_fields
        }
        trial_parity = [
            mlxs_trial["outputs"] == mlx_lm_trial["outputs"]
            for mlxs_trial, mlx_lm_trial in zip(
                mlxs_result["trials"],
                baseline["trials"],
                strict=True,
            )
        ]
        comparisons.append(
            {
                "prompt_target": prompt_target,
                "median_ratios": ratios,
                "exact_output_parity": all(trial_parity),
                "mlxs_summary": mlxs_result["summary"],
                "mlx_lm_summary": baseline["summary"],
            }
        )
    return comparisons
