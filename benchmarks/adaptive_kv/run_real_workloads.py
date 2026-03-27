"""CLI: real-repo adaptive KV workloads via ``generate()`` (Llama scope)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

from benchmarks.adaptive_kv.config_profiles import adaptive_config_for_baseline
from benchmarks.adaptive_kv.metrics_collect import (
    environment_fingerprint,
    estimated_full_kv_bytes_per_prompt_token,
    histogram_stats,
    sanitize_debug_snapshot,
    snapshot_in_memory_metrics,
)
from benchmarks.adaptive_kv.real_workloads import WORKLOAD_IDS, build_workload
from benchmarks.adaptive_kv.run import (
    SCHEMA_VERSION,
    _git_head,
    _pack_run,
    _preflight,
    _run_generate,
)

logger = logging.getLogger(__name__)

SOFT_SOFT = 42_000_000
SOFT_HARD = 250_000_000
HARD_SOFT = 30_000_000
HARD_HARD = 45_000_000


def _run_matrix(
    *,
    model: Any,
    tokenizer: Any,
    workloads: list[str],
    baselines: list[str],
    repeat: int,
    seed: int | None,
    prefill_step_size: int,
    clear_cache_interval: int,
    block_size_tokens: int,
    update_window_steps: int,
    soft_budget_bytes: int,
    hard_budget_bytes: int,
    suite_tag: str,
) -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[2]
    runs: list[dict[str, Any]] = []

    order = ("non_adaptive", "adaptive_full", "adaptive_soft", "adaptive_hard")
    bls = [b for b in order if b in baselines]

    for wl in workloads:
        for rep in range(repeat):
            run_seed = seed + rep if seed is not None else None
            sr = build_workload(wl, tokenizer, repo_root, run_seed=run_seed)
            ref_tokens: list[int] | None = None

            if "non_adaptive" in bls:
                if run_seed is not None:
                    mx.random.seed(run_seed)
                ids, _, _, elapsed = _run_generate(
                    model,
                    tokenizer,
                    sr.prompt_token_ids,
                    sr.options,
                    adaptive_config=None,
                    prefill_step_size=prefill_step_size,
                    clear_cache_interval=clear_cache_interval,
                )
                ref_tokens = ids
                n_gen = len(ids)
                total_tokens = len(sr.prompt_token_ids) + n_gen
                row = _pack_run(
                    scenario=wl,
                    baseline="non_adaptive",
                    repeat=rep,
                    scenario_result=sr,
                    token_ids=ids,
                    elapsed=elapsed,
                    metrics_snapshot=None,
                    adaptive_snapshots=[],
                    reference_match=None,
                    total_tokens=total_tokens,
                    kv_estimate_per_prompt_token=estimated_full_kv_bytes_per_prompt_token(model),
                )
                row["suite"] = suite_tag
                runs.append(row)

            for bl in bls:
                if bl == "non_adaptive":
                    continue
                acfg = adaptive_config_for_baseline(
                    bl,  # type: ignore[arg-type]
                    block_size_tokens=block_size_tokens,
                    update_window_steps=update_window_steps,
                    soft_budget_bytes=soft_budget_bytes,
                    hard_budget_bytes=hard_budget_bytes,
                    budget_profile="default",
                )
                if run_seed is not None:
                    mx.random.seed(run_seed)
                ids, metrics, snaps, elapsed = _run_generate(
                    model,
                    tokenizer,
                    sr.prompt_token_ids,
                    sr.options,
                    adaptive_config=acfg,
                    prefill_step_size=prefill_step_size,
                    clear_cache_interval=clear_cache_interval,
                )
                n_gen = len(ids)
                total_tokens = len(sr.prompt_token_ids) + n_gen
                mdict = snapshot_in_memory_metrics(metrics)
                policy_hist = mdict["histograms"].get("adaptive_kv_policy_time_seconds", [])
                match_ref: bool | None = None
                if ref_tokens is not None:
                    match_ref = ids == ref_tokens
                row = _pack_run(
                    scenario=wl,
                    baseline=bl,
                    repeat=rep,
                    scenario_result=sr,
                    token_ids=ids,
                    elapsed=elapsed,
                    metrics_snapshot=mdict,
                    adaptive_snapshots=[sanitize_debug_snapshot(s) for s in snaps],
                    reference_match=match_ref,
                    total_tokens=total_tokens,
                    kv_estimate_per_prompt_token=estimated_full_kv_bytes_per_prompt_token(model),
                    adaptive_config_dump=acfg.model_dump() if acfg is not None else None,
                    policy_time_stats=histogram_stats(policy_hist),
                )
                row["suite"] = suite_tag
                runs.append(row)

    return runs


def main() -> None:
    p = argparse.ArgumentParser(description="MLXs adaptive KV real workload evaluation")
    p.add_argument("--model", type=str, required=True)
    p.add_argument(
        "--workload",
        action="append",
        default=[],
        help="Workload id (repeatable) or omit for all",
    )
    p.add_argument(
        "--baseline",
        nargs="+",
        default=["non_adaptive", "adaptive_full", "adaptive_soft"],
        choices=["non_adaptive", "adaptive_full", "adaptive_soft", "adaptive_hard"],
    )
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefill-step-size", type=int, default=2048)
    p.add_argument("--clear-cache-interval", type=int, default=256)
    p.add_argument("--block-size-tokens", type=int, default=64)
    p.add_argument("--update-window-steps", type=int, default=16)
    p.add_argument("--soft-budget-bytes", type=int, default=SOFT_SOFT)
    p.add_argument("--hard-budget-bytes", type=int, default=SOFT_HARD)
    p.add_argument("--with-hard-subset", action="store_true")
    p.add_argument("-o", "--output", type=str, default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    model_path = args.model
    if not Path(model_path).exists():
        logger.error("Model path missing: %s", model_path)
        sys.exit(2)

    if "non_adaptive" not in args.baseline:
        p.error("non_adaptive is required for reference_token_match")

    wl_ids = list(WORKLOAD_IDS) if not args.workload else args.workload
    for w in wl_ids:
        if w not in WORKLOAD_IDS:
            p.error(f"unknown workload {w!r}")

    from mlxs.config.schema import ModelConfig
    from mlxs.load.formats import load_model_and_tokenizer

    mc = ModelConfig(
        model_path=model_path,
        trust_remote_code=args.trust_remote_code,
        preload=True,
        lazy_load=False,
    )
    model, tokenizer = load_model_and_tokenizer(model_path, mc)
    pre = _preflight(model)
    if not pre["supported"]:
        logger.error("Preflight failed: %s", pre.get("reason"))
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[2]
    all_runs: list[dict[str, Any]] = []

    all_runs.extend(
        _run_matrix(
            model=model,
            tokenizer=tokenizer,
            workloads=wl_ids,
            baselines=list(args.baseline),
            repeat=args.repeat,
            seed=args.seed,
            prefill_step_size=args.prefill_step_size,
            clear_cache_interval=args.clear_cache_interval,
            block_size_tokens=args.block_size_tokens,
            update_window_steps=args.update_window_steps,
            soft_budget_bytes=args.soft_budget_bytes,
            hard_budget_bytes=args.hard_budget_bytes,
            suite_tag="main_soft_budgets",
        )
    )

    if args.with_hard_subset:
        all_runs.extend(
            _run_matrix(
                model=model,
                tokenizer=tokenizer,
                workloads=["C2", "T1"],
                baselines=["non_adaptive", "adaptive_full", "adaptive_hard"],
                repeat=args.repeat,
                seed=args.seed,
                prefill_step_size=args.prefill_step_size,
                clear_cache_interval=args.clear_cache_interval,
                block_size_tokens=args.block_size_tokens,
                update_window_steps=args.update_window_steps,
                soft_budget_bytes=HARD_SOFT,
                hard_budget_bytes=HARD_HARD,
                suite_tag="hard_budget_eval",
            )
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "suite": "real_workloads_v1",
        "preflight": pre,
        "environment": environment_fingerprint(),
        "git_revision": _git_head(repo_root),
        "args": vars(args),
        "runs": all_runs,
    }

    out = args.output
    if out:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s", outp)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
