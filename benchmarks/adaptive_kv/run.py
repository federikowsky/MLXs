"""CLI runner: adaptive KV benchmarks via real ``generate()`` (compatibility-gated; Llama is the principal doc benchmark surface)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from benchmarks.adaptive_kv.config_profiles import BaselineKind, adaptive_config_for_baseline
from benchmarks.adaptive_kv.metrics_collect import (
    _rusage_maxrss_bytes,
    environment_fingerprint,
    estimated_full_kv_bytes_per_prompt_token,
    histogram_stats,
    sanitize_debug_snapshot,
    snapshot_in_memory_metrics,
)
from benchmarks.adaptive_kv.scenarios import SCENARIO_NAMES, build_scenario
from mlxs.generate import generate
from mlxs.observability.metrics import InMemoryMetrics

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _git_head(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _preflight(model: Any) -> dict[str, Any]:
    from mlxs.adaptive_kv.compatibility import assess_generation_compatibility

    comp = assess_generation_compatibility(
        model,
        cache=None,
        compile_decode=False,
        quantized_kv_start=0,
        input_embeddings_present=False,
    )
    args = getattr(model, "args", None)
    arch: dict[str, Any] = {}
    if args is not None:
        for name in (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "head_dim",
            "vocab_size",
        ):
            if hasattr(args, name):
                arch[name] = getattr(args, name)
    return {
        "supported": comp.supported,
        "reason": comp.reason,
        "num_layers": comp.num_layers,
        "model_type": getattr(model, "model_type", None),
        "architecture": arch,
    }


def _run_generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    options: Any,
    *,
    adaptive_config: Any | None,
    prefill_step_size: int,
    clear_cache_interval: int,
) -> tuple[list[int], InMemoryMetrics, list[dict[str, Any]], float]:
    metrics = InMemoryMetrics()
    final_adaptive: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {
        "prefill_step_size": prefill_step_size,
        "compile_decode": False,
        "clear_cache_interval": clear_cache_interval,
        "quantized_kv_start": 0,
        "metrics": metrics,
    }
    if adaptive_config is not None and adaptive_config.enabled:
        kwargs["adaptive_config"] = adaptive_config
        kwargs["final_adaptive_state_out"] = final_adaptive

    t0 = time.perf_counter()
    events = list(
        generate(
            model,
            tokenizer,
            prompt_ids,
            options,
            **kwargs,
        )
    )
    elapsed = time.perf_counter() - t0
    token_ids = [e.token_id for e in events]
    return token_ids, metrics, final_adaptive, elapsed


def _run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    from mlxs.config.schema import ModelConfig
    from mlxs.load.formats import load_model_and_tokenizer

    model_path = args.model
    model_cfg = ModelConfig(
        model_path=model_path,
        trust_remote_code=args.trust_remote_code,
        preload=True,
        lazy_load=False,
    )
    logger.info("Loading model from %s", model_path)
    model, tokenizer = load_model_and_tokenizer(model_path, model_cfg)

    pre = _preflight(model)
    if not pre["supported"]:
        raise SystemExit(f"adaptive_kv preflight failed: {pre['reason']}")

    scenarios = list(SCENARIO_NAMES) if "all" in args.scenario else list(args.scenario)
    order = ("non_adaptive", "adaptive_full", "adaptive_soft", "adaptive_hard")
    baselines: list[BaselineKind] = [b for b in order if b in args.baseline]  # type: ignore[misc]

    runs: list[dict[str, Any]] = []
    repo_root = Path(__file__).resolve().parents[2]

    for scenario_name in scenarios:
        ref_tokens: list[int] | None = None
        if "non_adaptive" in baselines:
            sr = build_scenario(
                scenario_name,
                tokenizer,
                seed=args.seed,
                prompt_target_tokens=args.prompt_target_tokens,
                decode_tokens=args.decode_tokens,
            )
            if args.seed is not None:
                mx.random.seed(args.seed)
            ids, _, _, elapsed = _run_generate(
                model,
                tokenizer,
                sr.prompt_token_ids,
                sr.options,
                adaptive_config=None,
                prefill_step_size=args.prefill_step_size,
                clear_cache_interval=args.clear_cache_interval,
            )
            n_gen = len(ids)
            total_tokens = len(sr.prompt_token_ids) + n_gen
            runs.append(
                _pack_run(
                    scenario=scenario_name,
                    baseline="non_adaptive",
                    repeat=0,
                    scenario_result=sr,
                    token_ids=ids,
                    elapsed=elapsed,
                    metrics_snapshot=None,
                    adaptive_snapshots=[],
                    reference_match=None,
                    total_tokens=total_tokens,
                    kv_estimate_per_prompt_token=estimated_full_kv_bytes_per_prompt_token(model),
                )
            )
            ref_tokens = ids

        for bl in baselines:
            if bl == "non_adaptive":
                continue
            for rep in range(args.repeat):
                sr = build_scenario(
                    scenario_name,
                    tokenizer,
                    seed=args.seed,
                    prompt_target_tokens=args.prompt_target_tokens,
                    decode_tokens=args.decode_tokens,
                )
                acfg = adaptive_config_for_baseline(
                    bl,
                    block_size_tokens=args.block_size_tokens,
                    update_window_steps=args.update_window_steps,
                    soft_budget_bytes=args.soft_budget_bytes,
                    hard_budget_bytes=args.hard_budget_bytes,
                    budget_profile=args.budget_profile,
                )
                if args.seed is not None:
                    mx.random.seed(args.seed)
                ids, metrics, snaps, elapsed = _run_generate(
                    model,
                    tokenizer,
                    sr.prompt_token_ids,
                    sr.options,
                    adaptive_config=acfg,
                    prefill_step_size=args.prefill_step_size,
                    clear_cache_interval=args.clear_cache_interval,
                )
                n_gen = len(ids)
                total_tokens = len(sr.prompt_token_ids) + n_gen
                mdict = snapshot_in_memory_metrics(metrics)
                policy_hist = mdict["histograms"].get("adaptive_kv_policy_time_seconds", [])
                match_ref: bool | None = None
                if ref_tokens is not None:
                    match_ref = ids == ref_tokens

                runs.append(
                    _pack_run(
                        scenario=scenario_name,
                        baseline=bl,
                        repeat=rep,
                        scenario_result=sr,
                        token_ids=ids,
                        elapsed=elapsed,
                        metrics_snapshot=mdict,
                        adaptive_snapshots=[sanitize_debug_snapshot(s) for s in snaps],
                        reference_match=match_ref,
                        total_tokens=total_tokens,
                        kv_estimate_per_prompt_token=estimated_full_kv_bytes_per_prompt_token(
                            model
                        ),
                        adaptive_config_dump=acfg.model_dump() if acfg is not None else None,
                        policy_time_stats=histogram_stats(policy_hist),
                    )
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "preflight": pre,
        "environment": environment_fingerprint(),
        "git_revision": _git_head(repo_root),
        "args": {
            "model": model_path,
            "scenario": scenarios,
            "baselines": baselines,
            "repeat": args.repeat,
            "seed": args.seed,
            "prompt_target_tokens": args.prompt_target_tokens,
            "decode_tokens": args.decode_tokens,
            "prefill_step_size": args.prefill_step_size,
            "clear_cache_interval": args.clear_cache_interval,
            "block_size_tokens": args.block_size_tokens,
            "update_window_steps": args.update_window_steps,
            "budget_profile": args.budget_profile,
            "soft_budget_bytes": args.soft_budget_bytes,
            "hard_budget_bytes": args.hard_budget_bytes,
        },
        "notes": {
            "recomputations_semantics": (
                "adaptive_kv_recomputations_total counts recompute requests, not replayed tokens."
            ),
        },
        "runs": runs,
    }


def _pack_run(
    *,
    scenario: str,
    baseline: str,
    repeat: int,
    scenario_result: Any,
    token_ids: list[int],
    elapsed: float,
    metrics_snapshot: dict[str, Any] | None,
    adaptive_snapshots: list[dict[str, Any]],
    reference_match: bool | None,
    total_tokens: int,
    kv_estimate_per_prompt_token: int | None,
    adaptive_config_dump: dict[str, Any] | None = None,
    policy_time_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    row: dict[str, Any] = {
        "scenario": scenario,
        "baseline": baseline,
        "repeat": repeat,
        "description": scenario_result.description,
        "scenario_meta": scenario_result.meta,
        "prompt_tokens": len(scenario_result.prompt_token_ids),
        "generated_tokens": len(token_ids),
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tps,
        "seconds_per_token": elapsed / total_tokens if total_tokens > 0 else None,
        "token_ids": token_ids,
        "reference_token_match": reference_match,
        "peak_rss_bytes_rusage": _rusage_maxrss_bytes(),
        "kv_bytes_per_prompt_token_estimate": kv_estimate_per_prompt_token,
    }
    if adaptive_config_dump is not None:
        row["adaptive_config"] = adaptive_config_dump
    if metrics_snapshot is not None:
        row["metrics"] = metrics_snapshot
    if adaptive_snapshots:
        row["adaptive_final_state"] = adaptive_snapshots[-1]
        if adaptive_snapshots[-1].get("resident_bytes") is not None:
            row["final_resident_bytes"] = adaptive_snapshots[-1]["resident_bytes"]
    if policy_time_stats is not None:
        row["policy_time_seconds_stats"] = policy_time_stats
    return row


def write_summary_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Adaptive KV benchmark summary\n",
        f"schema_version: {payload.get('schema_version')}\n",
        f"git_revision: {payload.get('git_revision')}\n",
        "",
        "| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |",
        "|----------|----------|-------|-------|---------|--------|-------|-------|-----------|",
    ]
    for r in payload.get("runs", []):
        m = r.get("metrics") or {}
        ctr = m.get("counters") or {}
        res = r.get("final_resident_bytes", "")
        row_fmt = (
            "| {scenario} | {baseline} | {tps:.2f} | {spt} | {gen} | {ref} | {res} | {ev} | {rc} |"
        )
        lines.append(
            row_fmt.format(
                scenario=r.get("scenario", ""),
                baseline=r.get("baseline", ""),
                tps=float(r.get("tokens_per_second") or 0),
                spt=f"{r.get('seconds_per_token'):.6f}" if r.get("seconds_per_token") else "",
                gen=r.get("generated_tokens", ""),
                ref=r.get("reference_token_match", ""),
                res=res,
                ev=int(ctr.get("adaptive_kv_evictions_total", 0)),
                rc=int(ctr.get("adaptive_kv_recomputations_total", 0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLXs adaptive KV benchmark harness")
    parser.add_argument("--model", type=str, default="", help="Local model dir or HF id")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Load model, print compatibility JSON, exit (no scenarios).",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["smoke"],
        help=f"Scenario name(s) or 'all'. Choices: {', '.join(SCENARIO_NAMES)}, all",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=["non_adaptive", "adaptive_full"],
        choices=["non_adaptive", "adaptive_full", "adaptive_soft", "adaptive_hard"],
        help="Baselines to run (plan section 3).",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-target-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--clear-cache-interval", type=int, default=256)
    parser.add_argument("--block-size-tokens", type=int, default=64)
    parser.add_argument("--update-window-steps", type=int, default=16)
    parser.add_argument(
        "--budget-profile",
        type=str,
        default="default",
        choices=["default", "tiny_stress"],
        help=(
            "Preset byte budgets for adaptive_soft / adaptive_hard; "
            "override with --soft-budget-bytes/--hard-budget-bytes."
        ),
    )
    parser.add_argument("--soft-budget-bytes", type=int, default=None)
    parser.add_argument("--hard-budget-bytes", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default="", help="Write JSON results path")
    parser.add_argument("--summary-md", type=str, default="", help="Write Markdown summary path")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    ns = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if ns.verbose else logging.INFO)

    if ns.repeat < 1:
        parser.error("--repeat must be >= 1")
    if ns.prompt_target_tokens < 8:
        parser.error("--prompt-target-tokens must be >= 8")
    if ns.decode_tokens < 1:
        parser.error("--decode-tokens must be >= 1")
    if ns.prefill_step_size < 1:
        parser.error("--prefill-step-size must be >= 1")
    if ns.clear_cache_interval < 0:
        parser.error("--clear-cache-interval must be >= 0")
    if ns.block_size_tokens < 1:
        parser.error("--block-size-tokens must be >= 1")
    if ns.update_window_steps < 1:
        parser.error("--update-window-steps must be >= 1")

    if not ns.model:
        parser.error("--model is required")

    if ns.preflight:
        from mlxs.config.schema import ModelConfig
        from mlxs.load.formats import load_model_and_tokenizer

        mc = ModelConfig(
            model_path=ns.model,
            trust_remote_code=ns.trust_remote_code,
            preload=True,
            lazy_load=False,
        )
        model, _tok = load_model_and_tokenizer(ns.model, mc)
        info = _preflight(model)
        print(json.dumps(info, indent=2))
        if not info.get("supported"):
            sys.exit(1)
        return

    payload = _run_matrix(ns)
    out = ns.output or ""
    if out:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s", outp)
    else:
        print(json.dumps(payload, indent=2))

    if ns.summary_md:
        write_summary_md(payload, Path(ns.summary_md))
        logger.info("Wrote %s", ns.summary_md)


if __name__ == "__main__":
    main()
