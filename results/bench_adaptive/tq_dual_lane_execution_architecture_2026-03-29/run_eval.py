from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from benchmarks.adaptive_kv.metrics_collect import (
    sanitize_debug_snapshot,
    snapshot_in_memory_metrics,
)

from mlxs._types import GenerateOptions
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.generate import generate
from mlxs.models.llama import Model as LlamaModel
from mlxs.models.llama import ModelArgs as LlamaModelArgs
from mlxs.models.ministral3 import Model as Ministral3Model
from mlxs.models.ministral3 import ModelArgs as Ministral3ModelArgs
from mlxs.models.qwen3_5 import Model as Qwen35Model
from mlxs.models.qwen3_5 import ModelArgs as Qwen35ModelArgs
from mlxs.observability.metrics import InMemoryMetrics

ROOT = Path(__file__).resolve().parents[3]
REPRESENTATIVE_FILE = (
    ROOT
    / "results/bench_adaptive"
    / "tq_execution_substrate_redesign_2026-03-28"
    / "representative_probes.json"
)
COMFORTABLE_FILE = (
    ROOT
    / "results/bench_adaptive"
    / "tq_execution_substrate_redesign_2026-03-28"
    / "comfortable_sweep.json"
)
MATRIX_FILE = (
    ROOT / "results/bench_adaptive" / "tq_final_hardening_proof_2026-03-28" / "matrix.json"
)
OUTPUT_FILE = Path(__file__).with_name("validation.json")

COMFORTABLE_CASES = (
    "llama_short_soft",
    "ministral3_short_soft_window",
    "qwen35_short_soft_hybrid",
)
REPRESENTATIVE_CASES = (
    "llama_short_hard",
    "llama_long_hard",
    "ministral3_short_hard",
    "ministral3_long_soft_96",
    "qwen35_short_hard_hybrid",
    "qwen35_long_soft_96",
)
REGIME_BY_CASE = {
    "llama_short_soft": "comfortable",
    "ministral3_short_soft_window": "comfortable",
    "qwen35_short_soft_hybrid": "comfortable",
    "llama_short_hard": "stressed",
    "llama_long_hard": "stressed",
    "ministral3_short_hard": "stressed",
    "ministral3_long_soft_96": "borderline",
    "qwen35_short_hard_hybrid": "stressed",
    "qwen35_long_soft_96": "borderline",
}
SUPPLEMENTAL_DORMANT_CASES = (
    "llama_short_soft",
    "llama_long_soft_96",
    "ministral3_short_soft_window",
    "ministral3_long_soft_96",
    "qwen35_short_soft_hybrid",
    "qwen35_long_soft_96",
)


class _NoStopTokenizer:
    eos_token_id = 999_999

    def encode(self, text: str) -> list[int]:
        del text
        return [1, 2, 3, 4]

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return chr(ord("a") + token_ids % 26)
        return "".join(chr(ord("a") + token % 26) for token in token_ids)


def _build_model(family: str, seed: int) -> Any:
    mx.random.seed(seed)
    if family == "family_a":
        return LlamaModel(
            LlamaModelArgs(
                hidden_size=128,
                num_hidden_layers=4,
                intermediate_size=256,
                num_attention_heads=4,
                num_key_value_heads=4,
                vocab_size=128,
                head_dim=32,
                layer_types=["full_attention"] * 4,
            )
        )
    if family == "family_b":
        return Ministral3Model(
            Ministral3ModelArgs(
                hidden_size=128,
                num_hidden_layers=4,
                intermediate_size=256,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=32,
                vocab_size=128,
                tie_word_embeddings=False,
                layer_types=[
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                ],
                sliding_window=64,
            )
        )
    if family == "family_c":
        return Qwen35Model(
            Qwen35ModelArgs(
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                intermediate_size=256,
                linear_num_value_heads=4,
                linear_num_key_heads=2,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_conv_kernel_dim=4,
                full_attention_interval=2,
                vocab_size=128,
                tie_word_embeddings=False,
            )
        )
    raise ValueError(f"Unsupported family {family!r}")


def _load_case_maps() -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    rep_data = json.loads(REPRESENTATIVE_FILE.read_text())
    comfort_data = json.loads(COMFORTABLE_FILE.read_text())
    matrix_data = json.loads(MATRIX_FILE.read_text())

    case_defs: dict[str, dict[str, Any]] = {}
    candidate_baselines: dict[str, float] = {}

    rep_cases = {row["case"]["name"]: row for row in rep_data["runs"]}
    for name in REPRESENTATIVE_CASES:
        row = rep_cases[name]
        case_defs[name] = row["case"]
        candidate_baselines[name] = float(row["summary"]["adaptive_tok_s"]["median"])

    core_cases = {row["case"]["name"]: row for row in matrix_data["core_cases"]}
    comfort_refs = {row["name"]: float(row["adaptive_median"]) for row in comfort_data}
    for name in COMFORTABLE_CASES:
        row = core_cases[name]
        case_defs[name] = row["case"]
        candidate_baselines[name] = comfort_refs[name]
    for name in SUPPLEMENTAL_DORMANT_CASES:
        if name in case_defs:
            continue
        row = core_cases.get(name)
        if row is not None:
            case_defs[name] = row["case"]

    return case_defs, candidate_baselines


def _run_case(
    case: dict[str, Any],
    *,
    adaptive: bool,
    tokenizer: Any,
) -> dict[str, Any]:
    model = _build_model(case["family"], case["seed"])
    prompt = case["prompt"]
    options = GenerateOptions(max_tokens=case["decode_tokens"], temperature=0.0, seed=case["seed"])
    kwargs: dict[str, Any] = {"prefill_step_size": case["prefill_step_size"]}
    metrics = InMemoryMetrics()
    final_state: list[dict[str, Any]] = []
    if adaptive:
        kwargs["adaptive_config"] = AdaptiveKVConfig(
            enabled=True,
            block_size_tokens=case["block_size_tokens"],
            update_window_steps=case["update_window_steps"],
            soft_budget_bytes=case["soft_budget_bytes"],
            hard_budget_bytes=case["hard_budget_bytes"],
            recent_tail_protect_blocks=0,
        )
        kwargs["metrics"] = metrics
        kwargs["final_adaptive_state_out"] = final_state

    started = time.perf_counter()
    events = list(generate(model, tokenizer, prompt, options, **kwargs))
    elapsed = time.perf_counter() - started
    mx.eval()
    if final_state:
        raw_snapshot = final_state[0]
        snapshot = sanitize_debug_snapshot(raw_snapshot)
        if "control_cadence" in raw_snapshot:
            snapshot["control_cadence"] = raw_snapshot["control_cadence"]
        if "hard_stabilization" in raw_snapshot:
            snapshot["hard_stabilization"] = raw_snapshot["hard_stabilization"]
    else:
        snapshot = None
    counters = snapshot_in_memory_metrics(metrics)["counters"] if adaptive else {}
    return {
        "token_ids": [event.token_id for event in events],
        "elapsed_s": elapsed,
        "tok_s": len(events) / elapsed if elapsed > 0 else 0.0,
        "snapshot": snapshot,
        "evictions": counters.get("adaptive_kv_evictions_total", 0),
        "recomputations": counters.get("adaptive_kv_recomputations_total", 0),
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    tokenizer = _NoStopTokenizer()
    case_defs, candidate_baselines = _load_case_maps()
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for name in (*COMFORTABLE_CASES, *REPRESENTATIVE_CASES):
        case = case_defs[name]
        mx.clear_cache()
        baseline = _run_case(case, adaptive=False, tokenizer=tokenizer)
        mx.clear_cache()
        _ = _run_case(case, adaptive=True, tokenizer=tokenizer)
        mx.clear_cache()
        measured: list[dict[str, Any]] = []
        for repeat in range(3):
            run = _run_case(case, adaptive=True, tokenizer=tokenizer)
            measured.append(
                {
                    "repeat": repeat,
                    "adaptive_tok_s": run["tok_s"],
                    "adaptive_elapsed_s": run["elapsed_s"],
                    "reference_token_match": run["token_ids"] == baseline["token_ids"],
                    "first_mismatch_token": next(
                        (
                            idx
                            for idx, (lhs, rhs) in enumerate(
                                zip(run["token_ids"], baseline["token_ids"], strict=False)
                            )
                            if lhs != rhs
                        ),
                        None,
                    ),
                    "evictions": run["evictions"],
                    "recomputations": run["recomputations"],
                    "snapshot": run["snapshot"],
                }
            )
            mx.clear_cache()

        tok_values = [row["adaptive_tok_s"] for row in measured]
        elapsed_values = [row["adaptive_elapsed_s"] for row in measured]
        pressure_states = sorted(
            {row["snapshot"]["pressure_state"] for row in measured if row["snapshot"] is not None}
        )
        resident_bytes = [
            row["snapshot"]["resident_bytes"] for row in measured if row["snapshot"] is not None
        ]
        control_windows = [
            row["snapshot"]["control_cadence"]["control_windows_total"]
            for row in measured
            if row["snapshot"] is not None
        ]
        usage_samples = [
            row["snapshot"]["control_cadence"]["usage_sample_steps_total"]
            for row in measured
            if row["snapshot"] is not None
        ]
        decode_steps = [
            row["snapshot"]["decode_steps"] for row in measured if row["snapshot"] is not None
        ]
        summary[name] = {
            "family": case["family"],
            "regime": REGIME_BY_CASE[name],
            "candidate_baseline_tok_s": candidate_baselines[name],
            "adaptive_tok_s": _stats(tok_values),
            "adaptive_elapsed_s": _stats(elapsed_values),
            "delta_vs_candidate_pct": (
                ((statistics.median(tok_values) / candidate_baselines[name]) - 1.0) * 100.0
            ),
            "reference_token_match_all": all(row["reference_token_match"] for row in measured),
            "first_mismatch_tokens": [row["first_mismatch_token"] for row in measured],
            "pressure_states": pressure_states,
            "resident_bytes": _stats([float(x) for x in resident_bytes]),
            "control_windows_total": _stats([float(x) for x in control_windows]),
            "usage_sample_steps_total": _stats([float(x) for x in usage_samples]),
            "decode_steps": _stats([float(x) for x in decode_steps]),
            "control_window_ratio": _stats(
                [float(w) / float(d) for w, d in zip(control_windows, decode_steps, strict=True)]
            ),
            "usage_sample_ratio": _stats(
                [float(s) / float(d) for s, d in zip(usage_samples, decode_steps, strict=True)]
            ),
            "observer_flushes_total": sorted(
                {
                    row["snapshot"]["attention_path"]["observer_flushes_total"]
                    for row in measured
                    if row["snapshot"] is not None
                }
            ),
            "n_execution_packs": sorted(
                {
                    row["snapshot"]["attention_path"]["n_execution_packs"]
                    for row in measured
                    if row["snapshot"] is not None
                }
            ),
            "n_visible_slices": sorted(
                {
                    row["snapshot"]["attention_path"]["n_visible_slices"]
                    for row in measured
                    if row["snapshot"] is not None
                }
            ),
            "evictions": _stats([float(row["evictions"]) for row in measured]),
            "recomputations": _stats([float(row["recomputations"]) for row in measured]),
        }
        rows.append(
            {
                "name": name,
                "case": case,
                "baseline_reference": {
                    "tok_s": baseline["tok_s"],
                    "elapsed_s": baseline["elapsed_s"],
                    "generated_tokens": len(baseline["token_ids"]),
                },
                "measured": measured,
                "summary": summary[name],
            }
        )

    dormant_rows: list[dict[str, Any]] = []
    dormant_summary: dict[str, Any] = {}
    for base_name in SUPPLEMENTAL_DORMANT_CASES:
        case = dict(case_defs[base_name])
        case["name"] = f"{base_name}_normal"
        case["soft_budget_bytes"] = 10**9
        case["hard_budget_bytes"] = 10**9

        mx.clear_cache()
        baseline = _run_case(case, adaptive=False, tokenizer=tokenizer)
        mx.clear_cache()
        _ = _run_case(case, adaptive=True, tokenizer=tokenizer)
        mx.clear_cache()
        measured: list[dict[str, Any]] = []
        for repeat in range(3):
            run = _run_case(case, adaptive=True, tokenizer=tokenizer)
            measured.append(
                {
                    "repeat": repeat,
                    "adaptive_tok_s": run["tok_s"],
                    "adaptive_elapsed_s": run["elapsed_s"],
                    "reference_token_match": run["token_ids"] == baseline["token_ids"],
                    "first_mismatch_token": next(
                        (
                            idx
                            for idx, (lhs, rhs) in enumerate(
                                zip(run["token_ids"], baseline["token_ids"], strict=False)
                            )
                            if lhs != rhs
                        ),
                        None,
                    ),
                    "evictions": run["evictions"],
                    "recomputations": run["recomputations"],
                    "snapshot": run["snapshot"],
                }
            )
            mx.clear_cache()

        tok_values = [row["adaptive_tok_s"] for row in measured]
        elapsed_values = [row["adaptive_elapsed_s"] for row in measured]
        control_windows = [
            row["snapshot"]["control_cadence"]["control_windows_total"]
            for row in measured
            if row["snapshot"] is not None
        ]
        usage_samples = [
            row["snapshot"]["control_cadence"]["usage_sample_steps_total"]
            for row in measured
            if row["snapshot"] is not None
        ]
        decode_steps = [
            row["snapshot"]["decode_steps"] for row in measured if row["snapshot"] is not None
        ]
        dormant_summary[case["name"]] = {
            "family": case["family"],
            "regime": "comfortable_normal",
            "baseline_tok_s": baseline["tok_s"],
            "adaptive_tok_s": _stats(tok_values),
            "adaptive_elapsed_s": _stats(elapsed_values),
            "delta_vs_non_adaptive_pct": (
                ((statistics.median(tok_values) / baseline["tok_s"]) - 1.0) * 100.0
            ),
            "reference_token_match_all": all(row["reference_token_match"] for row in measured),
            "pressure_states": sorted(
                {
                    row["snapshot"]["pressure_state"]
                    for row in measured
                    if row["snapshot"] is not None
                }
            ),
            "control_window_ratio": _stats(
                [float(w) / float(d) for w, d in zip(control_windows, decode_steps, strict=True)]
            ),
            "usage_sample_ratio": _stats(
                [float(s) / float(d) for s, d in zip(usage_samples, decode_steps, strict=True)]
            ),
        }
        dormant_rows.append(
            {
                "name": case["name"],
                "case": case,
                "baseline_reference": {
                    "tok_s": baseline["tok_s"],
                    "elapsed_s": baseline["elapsed_s"],
                    "generated_tokens": len(baseline["token_ids"]),
                },
                "measured": measured,
                "summary": dormant_summary[case["name"]],
            }
        )

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "rows": rows,
                "summary": summary,
                "dormant_rows": dormant_rows,
                "dormant_summary": dormant_summary,
            },
            indent=2,
        )
    )

    print(f"Wrote {OUTPUT_FILE}")
    for name in (*COMFORTABLE_CASES, *REPRESENTATIVE_CASES):
        entry = summary[name]
        print(
            f"{name}: median={entry['adaptive_tok_s']['median']:.2f} tok/s "
            f"delta={entry['delta_vs_candidate_pct']:+.2f}% "
            f"samples={entry['usage_sample_ratio']['median']:.2f} "
            f"controls={entry['control_window_ratio']['median']:.2f}"
        )
    for name in dormant_summary:
        entry = dormant_summary[name]
        print(
            f"{name}: median={entry['adaptive_tok_s']['median']:.2f} tok/s "
            f"vs_non_adaptive={entry['delta_vs_non_adaptive_pct']:+.2f}% "
            f"samples={entry['usage_sample_ratio']['median']:.2f} "
            f"controls={entry['control_window_ratio']['median']:.2f}"
        )


if __name__ == "__main__":
    main()
