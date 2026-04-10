"""Tests for the Phase 2 benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.mlxs_vs_mlx_lm.backends import GenerationMetrics, _decode_tokens_measured
from benchmarks.mlxs_vs_mlx_lm.discovery import LocalModelInfo
from benchmarks.mlxs_vs_mlx_lm.harness import (
    CANONICAL_MODE,
    HarnessConfig,
    _build_prompt_token_ids,
    local_model_from_path,
    run_harness,
)


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def test_build_prompt_token_ids_hits_exact_target() -> None:
    token_ids = _build_prompt_token_ids(_FakeTokenizer(), 12)
    assert len(token_ids) == 12
    assert token_ids == tuple(range(12))


def test_decode_tokens_measured_uses_generated_minus_one() -> None:
    assert _decode_tokens_measured(0) == 0
    assert _decode_tokens_measured(1) == 0
    assert _decode_tokens_measured(128) == 127


def test_local_model_from_path_records_weight_format_class(tmp_path: Path) -> None:
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "quantization": {"bits": 4, "group_size": 64},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    info = local_model_from_path(tmp_path)
    assert info is not None
    assert info.weight_format_class == "mlx_q4_g64"


def test_run_harness_uses_shared_prompt_ids_and_emits_phase2_metadata(monkeypatch) -> None:
    seen_prompt_ids: dict[str, tuple[int, ...]] = {}

    def fake_load_reference_tokenizer(model_path: Path, *, trust_remote_code: bool) -> _FakeTokenizer:
        del model_path, trust_remote_code
        return _FakeTokenizer()

    def fake_run_backend_session(
        backend_variant: str,
        model_path: Path,
        prompt_token_ids: tuple[int, ...],
        cfg: HarnessConfig,
    ) -> tuple[dict[str, object], list[GenerationMetrics]]:
        del model_path
        seen_prompt_ids[backend_variant] = prompt_token_ids
        metrics = GenerationMetrics(
            backend="mlx_lm" if backend_variant == "mlx_lm" else "mlxs",
            variant=backend_variant,
            prompt_tokens=len(prompt_token_ids),
            generated_tokens=cfg.max_tokens,
            decode_tokens_measured=cfg.max_tokens - 1,
            prefill_wall_s=0.2,
            ttft_s=0.3,
            decode_wall_s=1.0,
            end_to_end_wall_s=1.3,
            prefill_tok_per_s=100.0,
            decode_tok_per_s=127.0,
            end_to_end_tok_per_s=98.0,
            rss_bytes_after_generate=123,
            mlx_peak_memory_bytes=456,
            error=None,
        )
        header = {
            "backend": "mlx_lm" if backend_variant == "mlx_lm" else "mlxs",
            "variant": backend_variant,
            "load_wall_s": 0.1,
            "load_error": None,
        }
        return header, [metrics]

    monkeypatch.setattr(
        "benchmarks.mlxs_vs_mlx_lm.harness._load_reference_tokenizer",
        fake_load_reference_tokenizer,
    )
    monkeypatch.setattr(
        "benchmarks.mlxs_vs_mlx_lm.harness._run_backend_session",
        fake_run_backend_session,
    )

    model_info = LocalModelInfo(
        path=Path("/tmp/fake-model"),
        hub_repo_id="mlx-community/fake-model",
        model_type="llama",
        architectures=("LlamaForCausalLM",),
        safetensors_files=("model.safetensors",),
        weight_bytes=1024,
        weight_format_class="mlx_q4_g64",
    )
    cfg = HarnessConfig(
        mode=CANONICAL_MODE,
        warmup_runs=2,
        timed_runs=1,
        max_tokens=128,
        seed=0,
        prefill_step_size=2048,
        trust_remote_code=False,
        prompt_targets=(16,),
        backends=frozenset({"mlxs", "mlx_lm"}),
        run_compiled_secondary=True,
    )

    payload = run_harness([model_info], cfg)
    row = payload["results"][0]

    assert payload["benchmark"]["benchmark_class"] == "Class A — Minimal fast-path decode"
    assert payload["benchmark"]["diagnostics_disabled"] is True
    assert "prefill" in payload["measurement_boundaries"]
    assert "layer4" in payload["contamination_exclusions"]
    assert row["prompt_input_policy"] == "shared_pre_tokenized_ids"
    assert row["decode_target_tokens"] == 128
    assert row["finish_condition_policy"] == "fixed_max_tokens_only"
    assert seen_prompt_ids["mlx_lm"] == seen_prompt_ids["mlxs_eager"]
    assert seen_prompt_ids["mlx_lm"] == seen_prompt_ids["mlxs_compiled"]
    assert "mlxs_eager_over_mlx_lm_decode_tok_per_s_median_ratio" in row["comparison"]["median_ratios"]
