from __future__ import annotations

from benchmarks.mlxs_vs_mlx_lm.r30_ac2_substrate_probe import (
    _compare_tokens,
    _group_summary,
    _model_kind,
)


def test_model_kind_maps_supported_r30_models() -> None:
    assert _model_kind("mlx-community/Llama-3.2-1B-Instruct-4bit") == "llama"
    assert _model_kind("mlx-community/Llama-3.2-3B-Instruct-4bit") == "llama"
    assert _model_kind("mlx-community/Qwen2.5-1.5B-Instruct-4bit") == "qwen2"


def test_compare_tokens_reports_prefix_and_mismatch() -> None:
    result = _compare_tokens([1, 2, 3, 4], [1, 2, 9, 4], prefix=3)

    assert result["exact"] is False
    assert result["same_count"] is True
    assert result["prefix_match"] is False
    assert result["first_mismatch_index"] == 2
    assert result["left_prefix"] == [1, 2, 3]
    assert result["right_prefix"] == [1, 2, 9]


def test_group_summary_uses_medians_for_times_and_steps() -> None:
    summary = _group_summary(
        {"a": 0.5, "b": 1.0},
        {"a": 2.0, "b": 4.0},
        {"a": 2, "b": 6},
        {"a": 8, "b": 10},
        ["a", "b"],
    )

    assert summary["ttft_s"] == 0.75
    assert summary["completion_s"] == 3.0
    assert summary["first_token_step"] == 4.0
    assert summary["completion_step"] == 9.0
