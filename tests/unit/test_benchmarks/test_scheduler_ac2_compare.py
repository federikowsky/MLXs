from __future__ import annotations

from benchmarks.mlxs_vs_mlx_lm.scheduler_ac2_utils import (
    compare_prompt_results,
    summarize_probe_trials,
)


def test_summarize_probe_trials_reports_median_and_range() -> None:
    trials = [
        {
            "requests_per_s": 2.0,
            "generated_tok_per_s": 128.0,
            "p50_ttft_s": 0.5,
            "p95_completion_s": 1.5,
            "rss_bytes": 100.0,
            "mlx_peak_memory_bytes": 200.0,
        },
        {
            "requests_per_s": 4.0,
            "generated_tok_per_s": 256.0,
            "p50_ttft_s": 0.75,
            "p95_completion_s": 1.75,
            "rss_bytes": 120.0,
            "mlx_peak_memory_bytes": 220.0,
        },
    ]

    summary = summarize_probe_trials(trials)

    assert summary["requests_per_s"]["median"] == 3.0
    assert summary["p50_ttft_s"]["min"] == 0.5
    assert summary["mlx_peak_memory_bytes"]["max"] == 220.0


def test_compare_prompt_results_reports_ratios_and_output_parity() -> None:
    mlxs_results = [
        {
            "prompt_target": 256,
            "trials": [{"outputs": {"0": [1, 2], "1": [1, 2]}}],
            "summary": summarize_probe_trials(
                [
                    {
                        "requests_per_s": 2.0,
                        "generated_tok_per_s": 128.0,
                        "p50_ttft_s": 0.5,
                        "p95_completion_s": 1.5,
                        "rss_bytes": 100.0,
                        "mlx_peak_memory_bytes": 200.0,
                    }
                ]
            ),
        }
    ]
    mlx_lm_results = [
        {
            "prompt_target": 256,
            "trials": [{"outputs": {"0": [1, 2], "1": [1, 2]}}],
            "summary": summarize_probe_trials(
                [
                    {
                        "requests_per_s": 1.0,
                        "generated_tok_per_s": 64.0,
                        "p50_ttft_s": 1.0,
                        "p95_completion_s": 3.0,
                        "rss_bytes": 120.0,
                        "mlx_peak_memory_bytes": 240.0,
                    }
                ]
            ),
        }
    ]

    comparisons = compare_prompt_results(mlxs_results, mlx_lm_results)

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["prompt_target"] == 256
    assert comparison["exact_output_parity"] is True
    assert comparison["median_ratios"]["mlxs_over_mlx_lm_requests_per_s_median_ratio"] == 2.0
    assert comparison["median_ratios"]["mlxs_over_mlx_lm_p50_ttft_s_median_ratio"] == 0.5
