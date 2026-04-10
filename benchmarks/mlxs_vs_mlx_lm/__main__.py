"""CLI for canonical Class A and exploratory MLXs vs mlx-lm benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.mlxs_vs_mlx_lm.discovery import discover_local_models, parse_explicit_model_paths
from benchmarks.mlxs_vs_mlx_lm.harness import (
    CANONICAL_MAX_TOKENS,
    CANONICAL_MODE,
    CANONICAL_PROMPT_TARGETS,
    CANONICAL_TIMED_RUNS,
    CANONICAL_WARMUP_RUNS,
    EXPLORATORY_MODE,
)
from benchmarks.mlxs_vs_mlx_lm.json_util import sanitize_for_json


def _parse_targets(s: str) -> tuple[int, ...]:
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("need at least one prompt length")
    return tuple(parts)


def _parse_backends(s: str) -> frozenset[str]:
    raw = {x.strip().lower() for x in s.split(",") if x.strip()}
    allowed = frozenset({"mlxs", "mlx_lm", "both"})
    if not raw <= allowed:
        bad = raw - allowed
        raise argparse.ArgumentTypeError(f"unknown backend(s): {bad}")
    if "both" in raw or not raw:
        return frozenset({"mlxs", "mlx_lm"})
    return frozenset(raw)


def _missing_runtime_dependencies(backends: frozenset[str]) -> list[str]:
    required = ["transformers", "mlx"]
    if "mlx_lm" in backends:
        required.append("mlx_lm")

    missing: list[str] = []
    for module_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def _print_text_summary(payload: dict) -> None:
    benchmark = payload.get("benchmark", {})
    print(f"\n=== {benchmark.get('benchmark_class', 'Benchmark')} summary ===\n")
    print(f"mode: {benchmark.get('benchmark_mode', 'unknown')}")
    print(f"warmup: {benchmark.get('warmup_rule', 'n/a')}")
    print(f"aggregation: {benchmark.get('aggregation_rule', 'n/a')}\n")

    for row in payload.get("results", []):
        if row.get("skipped"):
            print(f"[skip] {row.get('model_hub_id')}: {row.get('skip_reason')}")
            continue

        model_id = row["model_hub_id"]
        prompt_tokens = row["prompt_token_count"]
        decode_target = row["decode_target_tokens"]
        print(
            f"{model_id}  |  prompt={prompt_tokens} tok  |  decode={decode_target} tok  "
            f"|  {row['weight_format_class']}"
        )
        for key in ("mlx_lm", "mlxs_eager", "mlxs_compiled"):
            if key not in row:
                continue
            session = row[key].get("session", {})
            load_error = session.get("load_error")
            if load_error:
                print(f"  {key}: load failed: {load_error}")
                continue
            stats = row[key]["stats"]
            decode = stats.get("decode_tok_per_s", {})
            prefill = stats.get("prefill_tok_per_s", {})
            ttft = stats.get("ttft_s", {})
            e2e = stats.get("end_to_end_wall_s", {})
            print(
                f"  {key}: load {session.get('load_wall_s', 0):.2f}s  "
                f"prefill {prefill.get('median', float('nan')):.2f} tok/s  "
                f"decode {decode.get('median', float('nan')):.2f} tok/s  "
                f"TTFT {ttft.get('median', float('nan')) * 1000:.2f} ms  "
                f"e2e {e2e.get('median', float('nan')):.3f} s"
            )
        if "comparison" in row:
            for metric, ratio in row["comparison"]["median_ratios"].items():
                print(f"  ratio {metric}: {ratio:.4f}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Phase 2 Class A benchmark: MLXs eager primary path vs mlx-lm, "
            "with compiled MLXs as a separately labeled secondary variant."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(CANONICAL_MODE, EXPLORATORY_MODE),
        default=CANONICAL_MODE,
        help="`canonical` runs the frozen Phase 2 Class A sign-off case; `exploratory` keeps hub scanning.",
    )
    parser.add_argument(
        "--hub-root",
        type=Path,
        default=None,
        help="Hugging Face hub cache root (used only in exploratory mode).",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        default="",
        help="Comma-separated snapshot directories (exploratory mode only).",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=6,
        help="When scanning the hub in exploratory mode: keep N smallest checkpoints by safetensors size.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print discovered causal-LM snapshots and exit.",
    )
    parser.add_argument(
        "--prompt-targets",
        type=_parse_targets,
        default=CANONICAL_PROMPT_TARGETS,
        help="Comma-separated target prompt lengths (exploratory mode only).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=CANONICAL_MAX_TOKENS,
        help="Generated tokens per timed trial.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=CANONICAL_WARMUP_RUNS,
        help="Warmup generations per backend and prompt.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=CANONICAL_TIMED_RUNS,
        help="Timed generations per backend and prompt.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base PRNG seed (incremented per run).")
    parser.add_argument(
        "--prefill-step",
        type=int,
        default=2048,
        help="Chunked prefill step size for both canonical paths.",
    )
    parser.add_argument(
        "--no-compile-decode",
        action="store_true",
        help="Skip the separately labeled compiled MLXs secondary variant.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward to tokenizer loaders.",
    )
    parser.add_argument(
        "--backends",
        type=_parse_backends,
        default=frozenset({"mlxs", "mlx_lm"}),
        help="mlx_lm, mlxs, or both.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the full JSON report.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a human-readable summary to stdout.",
    )
    args = parser.parse_args(argv)

    if args.list_models:
        found = discover_local_models(args.hub_root, max_models=None)
        for model in found:
            mb = model.weight_bytes / (1024 * 1024)
            print(
                f"{model.weight_bytes:12d} B  {mb:8.1f} MiB  "
                f"{model.weight_format_class:16s}  {model.model_type:24s}  {model.path}"
            )
        print(f"\nTotal: {len(found)}")
        return 0

    missing = _missing_runtime_dependencies(args.backends)
    if missing:
        print(
            "Phase 2 blocked: missing benchmark dependencies in the active interpreter: "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
        return 3

    from benchmarks.mlxs_vs_mlx_lm.harness import HarnessConfig, resolve_models, run_harness

    explicit_paths = parse_explicit_model_paths(args.model_paths) if args.model_paths.strip() else None
    prompt_targets = CANONICAL_PROMPT_TARGETS if args.mode == CANONICAL_MODE else args.prompt_targets
    max_tokens = CANONICAL_MAX_TOKENS if args.mode == CANONICAL_MODE else max(1, args.max_tokens)
    warmup_runs = CANONICAL_WARMUP_RUNS if args.mode == CANONICAL_MODE else max(0, args.warmup)
    timed_runs = CANONICAL_TIMED_RUNS if args.mode == CANONICAL_MODE else max(1, args.runs)

    models = resolve_models(
        hub_root=args.hub_root,
        max_models=None if explicit_paths else args.max_models,
        explicit_paths=explicit_paths,
        mode=args.mode,
    )
    if not models:
        if args.mode == CANONICAL_MODE:
            print("Phase 2 blocked: canonical Class A model path is missing or invalid.", file=sys.stderr)
        else:
            print("No models found. Use --list-models or --model-paths.", file=sys.stderr)
        return 2

    cfg = HarnessConfig(
        mode=args.mode,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        max_tokens=max_tokens,
        seed=args.seed,
        prefill_step_size=max(1, args.prefill_step),
        trust_remote_code=args.trust_remote_code,
        prompt_targets=prompt_targets,
        backends=args.backends,
        run_compiled_secondary=not args.no_compile_decode,
    )

    payload = run_harness(models, cfg)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        safe = sanitize_for_json(payload)
        args.output.write_text(
            json.dumps(safe, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"Wrote {args.output}", file=sys.stderr)
    if args.print_summary or args.output is None:
        _print_text_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
