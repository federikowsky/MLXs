"""CLI: ``python -m benchmarks.mlxs_vs_mlx_lm`` (set ``PYTHONPATH`` to repo root)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.mlxs_vs_mlx_lm.discovery import discover_local_models, parse_explicit_model_paths
from benchmarks.mlxs_vs_mlx_lm.harness import HarnessConfig, resolve_models, run_harness
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
    if "both" in raw:
        return frozenset({"mlxs", "mlx_lm"})
    if not raw:
        return frozenset({"mlxs", "mlx_lm"})
    return frozenset(raw)


def _print_text_summary(payload: dict) -> None:
    print("\n=== MLXs vs mlx-lm — summary (median over timed runs) ===\n")
    for row in payload.get("results", []):
        if row.get("skipped"):
            print(f"[skip] {row.get('model_hub_id')}: {row.get('skip_reason')}")
            continue
        mid = row["model_hub_id"]
        pt = row["prompt_target_tokens"]
        print(f"\n{mid}  |  prompt≈{pt} tok  |  {row['model_type']}")
        if "mlx_lm" in row:
            sess = row["mlx_lm"].get("session", {})
            le = sess.get("load_error")
            if le:
                print(f"  mlx_lm: load failed: {le}")
            else:
                s = row["mlx_lm"]["stats"]
                d = s.get("decode_tok_per_s", {})
                p = s.get("prefill_effective_tok_per_s", {})
                t = s.get("ttft_s", {})
                rss = s.get("rss_bytes_after_generate", {})
                print(
                    f"  mlx_lm: load {sess.get('load_wall_s', 0):.2f}s  "
                    f"decode {d.get('median', float('nan')):.2f} tok/s  "
                    f"prefill_eff {p.get('median', float('nan')):.2f} tok/s  "
                    f"TTFT {t.get('median', float('nan')) * 1000:.2f} ms  "
                    f"RSS~{rss.get('median', 0) / 1e6:.0f} MB"
                )
        if "mlxs" in row:
            sess = row["mlxs"].get("session", {})
            le = sess.get("load_error")
            if le:
                print(f"  mlxs: load failed: {le}")
            else:
                s = row["mlxs"]["stats"]
                d = s.get("decode_tok_per_s", {})
                p = s.get("prefill_effective_tok_per_s", {})
                t = s.get("ttft_s", {})
                rss = s.get("rss_bytes_after_generate", {})
                print(
                    f"  mlxs:   load {sess.get('load_wall_s', 0):.2f}s  "
                    f"decode {d.get('median', float('nan')):.2f} tok/s  "
                    f"prefill_eff {p.get('median', float('nan')):.2f} tok/s  "
                    f"TTFT {t.get('median', float('nan')) * 1000:.2f} ms  "
                    f"RSS~{rss.get('median', 0) / 1e6:.0f} MB"
                )
        if "comparison" in row:
            for k, v in row["comparison"]["median_ratios"].items():
                print(f"  ratio {k}: {v:.4f} (>1 means MLXs faster on that metric)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Professional greedy-decoding benchmark: MLXs vs mlx-lm on local HF caches. "
            "Adaptive KV is not used. Default MLXs: compile_decode on, prefill_step 2048 "
            "(matches mlx-lm generate_step default). Requires: pip install mlx-lm."
        ),
    )
    p.add_argument(
        "--hub-root",
        type=Path,
        default=None,
        help="Hugging Face hub cache root (default: ~/.cache/huggingface/hub).",
    )
    p.add_argument(
        "--model-paths",
        type=str,
        default="",
        help="Comma-separated snapshot directories (overrides hub scan).",
    )
    p.add_argument(
        "--max-models",
        type=int,
        default=6,
        help="When scanning hub: keep N smallest checkpoints by safetensors size.",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print discovered causal-LM snapshots and exit.",
    )
    p.add_argument(
        "--prompt-targets",
        type=_parse_targets,
        default="256,2048",
        help="Comma-separated target prompt lengths (tokenizer tokens).",
    )
    p.add_argument("--max-tokens", type=int, default=128, help="Generated tokens per trial.")
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup generations per backend and prompt.",
    )
    p.add_argument("--runs", type=int, default=7, help="Timed generations (statistics).")
    p.add_argument("--seed", type=int, default=0, help="Base PRNG seed (incremented per run).")
    p.add_argument(
        "--prefill-step",
        type=int,
        default=2048,
        help="MLXs chunked prefill step (default 2048, same as mlx-lm generate_step).",
    )
    p.add_argument(
        "--no-compile-decode",
        action="store_true",
        help="Disable MLXs mx.compile on decode forward (default: compile enabled).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward to tokenizer loaders.",
    )
    p.add_argument(
        "--backends",
        type=_parse_backends,
        default="both",
        help="mlx_lm, mlxs, or both.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write full JSON report (all trials + order statistics).",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print human-readable median table to stdout.",
    )
    args = p.parse_args(argv)

    if args.list_models:
        found = discover_local_models(args.hub_root, max_models=None)
        for m in found:
            mb = m.weight_bytes / (1024 * 1024)
            print(f"{m.weight_bytes:12d} B  {mb:8.1f} MiB  {m.model_type:24s}  {m.path}")
        print(f"\nTotal: {len(found)}")
        return 0

    explicit = parse_explicit_model_paths(args.model_paths) if args.model_paths.strip() else None
    models = resolve_models(
        hub_root=args.hub_root,
        max_models=None if explicit else args.max_models,
        explicit_paths=explicit,
    )
    if not models:
        print("No models found. Use --list-models or --model-paths.", file=sys.stderr)
        return 2

    if "mlx_lm" in args.backends:
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            print("mlx_lm backend requested but mlx-lm is not installed.", file=sys.stderr)
            return 3

    cfg = HarnessConfig(
        warmup_runs=max(0, args.warmup),
        timed_runs=max(1, args.runs),
        max_tokens=max(1, args.max_tokens),
        seed=args.seed,
        prefill_step_size=max(1, args.prefill_step),
        compile_decode=not args.no_compile_decode,
        trust_remote_code=args.trust_remote_code,
        prompt_targets=args.prompt_targets,
        backends=args.backends,
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
