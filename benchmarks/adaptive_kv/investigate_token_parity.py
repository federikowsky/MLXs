"""One-off: token parity diff for real workloads C5 / T4 (debugging)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx

from benchmarks.adaptive_kv.config_profiles import adaptive_config_for_baseline
from benchmarks.adaptive_kv.metrics_collect import sanitize_debug_snapshot
from benchmarks.adaptive_kv.real_workloads import build_workload
from mlxs.generate import generate
from mlxs.observability.metrics import InMemoryMetrics


def _first_diff(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def _run(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    options: Any,
    *,
    adaptive_config: Any | None,
) -> tuple[list[int], dict[str, Any] | None, dict[str, int]]:
    metrics = InMemoryMetrics()
    final: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {
        "prefill_step_size": 2048,
        "compile_decode": False,
        "clear_cache_interval": 256,
        "quantized_kv_start": 0,
        "metrics": metrics,
    }
    if adaptive_config is not None and adaptive_config.enabled:
        kwargs["adaptive_config"] = adaptive_config
        kwargs["final_adaptive_state_out"] = final
    events = list(
        generate(model, tokenizer, prompt_ids, options, **kwargs),
    )
    ids = [e.token_id for e in events]
    snap = sanitize_debug_snapshot(final[-1]) if final else None
    ctr: dict[str, int] = {}
    with metrics._lock:
        ctr = {k: int(v) for k, v in metrics._counters.items() if k.startswith("adaptive_")}
    return ids, snap, ctr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--workload", type=str, required=True, choices=["C5", "T4"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-tokens-at-divergence", type=int, default=-1)
    p.add_argument("-o", type=str, default="")
    args = p.parse_args()

    from mlxs.config.schema import ModelConfig
    from mlxs.load.formats import load_model_and_tokenizer

    repo = Path(__file__).resolve().parents[2]
    mc = ModelConfig(model_path=args.model, preload=True, lazy_load=False)
    model, tokenizer = load_model_and_tokenizer(args.model, mc)

    out: dict[str, Any] = {"workload": args.workload, "seeds": [], "runs": []}

    for rep in range(args.repeats):
        run_seed = args.seed + rep
        sr = build_workload(args.workload, tokenizer, repo, run_seed=run_seed)
        base_opts = sr.options
        if args.max_tokens_at_divergence >= 0:
            base_opts = replace(sr.options, max_tokens=args.max_tokens_at_divergence)

        row: dict[str, Any] = {"run_seed": run_seed, "prompt_tokens": len(sr.prompt_token_ids)}
        mx.random.seed(run_seed)
        na_ids, _, _ = _run(model, tokenizer, sr.prompt_token_ids, base_opts, adaptive_config=None)
        row["non_adaptive_ids"] = na_ids

        for bl, soft, hard in (
            ("adaptive_full", None, None),
            ("adaptive_soft", 42_000_000, 250_000_000),
        ):
            cfg = adaptive_config_for_baseline(
                bl,  # type: ignore[arg-type]
                block_size_tokens=64,
                update_window_steps=16,
                soft_budget_bytes=soft,
                hard_budget_bytes=hard,
                budget_profile="default",
            )
            mx.random.seed(run_seed)
            ad_ids, snap, ctr = _run(
                model,
                tokenizer,
                sr.prompt_token_ids,
                base_opts,
                adaptive_config=cfg,
            )
            di = _first_diff(na_ids, ad_ids)
            row[bl] = {
                "ids": ad_ids,
                "first_diff_vs_na": di,
                "snapshot": snap,
                "counters": ctr,
            }
        out["seeds"].append(run_seed)
        out["runs"].append(row)

    if args.o:
        Path(args.o).parent.mkdir(parents=True, exist_ok=True)
        Path(args.o).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("wrote", args.o)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
