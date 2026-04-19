"""Scheduler-level AC2 probe for MLXs batch throughput redesign work."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from transformers import AutoTokenizer

from benchmarks.mlxs_vs_mlx_lm.harness import CANONICAL_MODEL_PATH, _build_prompt_token_ids
from benchmarks.mlxs_vs_mlx_lm.memory import rss_bytes_self
from mlxs._types import GenerateOptions, ModelMode
from mlxs.batch.scheduler import BatchScheduler
from mlxs.load.loader import load_model


def _p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _run_case(
    model: Any,
    tokenizer: Any,
    *,
    n: int,
    max_tokens: int,
    prompt_target: int,
    prefill_step_size: int,
    trial_seed: int,
) -> dict[str, Any]:
    prompt = _build_prompt_token_ids(tokenizer, prompt_target)
    scheduler = BatchScheduler(
        prefill_batch_size=n,
        completion_batch_size=n,
        prefill_step_size=prefill_step_size,
    )
    options = GenerateOptions(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=trial_seed,
        stop_sequences=(),
        extra_eos_token_ids=(),
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        stream=True,
    )

    for idx in range(n):
        scheduler.add(f"r{idx}", model, tokenizer, prompt, options)

    outputs = {f"r{idx}": [] for idx in range(n)}
    first_token_at: dict[str, float] = {}
    completion_at: dict[str, float] = {}

    mx.reset_peak_memory()
    mx.clear_cache()
    t0 = time.perf_counter()
    while scheduler.pending_count or scheduler.active_count:
        step_results = scheduler.step()
        now = time.perf_counter()
        for request_id, events in step_results.items():
            outputs[request_id].extend(int(event.token_id) for event in events)
            if request_id not in first_token_at and events:
                first_token_at[request_id] = now - t0
        for request_id, _events, _final_cache in scheduler.drain():
            completion_at[request_id] = now - t0

    wall = time.perf_counter() - t0
    generated_tokens = sum(len(tokens) for tokens in outputs.values())
    return {
        "prompt_target": prompt_target,
        "trial_seed": trial_seed,
        "requests_per_s": n / wall,
        "generated_tok_per_s": generated_tokens / wall,
        "p50_ttft_s": statistics.median(first_token_at.values()),
        "p95_completion_s": _p95(list(completion_at.values())),
        "rss_bytes": rss_bytes_self(),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "outputs": outputs,
    }


def _summarize_trials(trials: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = (
        "requests_per_s",
        "generated_tok_per_s",
        "p50_ttft_s",
        "p95_completion_s",
        "rss_bytes",
        "mlx_peak_memory_bytes",
    )
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [float(trial[field]) for trial in trials]
        summary[field] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }
    return summary


def run_probe(
    *,
    prompt_targets: tuple[int, ...],
    n: int,
    max_tokens: int,
    warmup_runs: int,
    timed_runs: int,
    prefill_step_size: int,
    trust_remote_code: bool,
    seed: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(CANONICAL_MODEL_PATH),
        trust_remote_code=trust_remote_code,
    )
    model = load_model(
        CANONICAL_MODEL_PATH,
        lazy=False,
        model_mode=ModelMode.AUTO,
    )

    results = []
    for prompt_target in prompt_targets:
        for warm_idx in range(warmup_runs):
            _run_case(
                model,
                tokenizer,
                n=n,
                max_tokens=max_tokens,
                prompt_target=prompt_target,
                prefill_step_size=prefill_step_size,
                trial_seed=seed - 1000 - warm_idx,
            )
        trials = [
            _run_case(
                model,
                tokenizer,
                n=n,
                max_tokens=max_tokens,
                prompt_target=prompt_target,
                prefill_step_size=prefill_step_size,
                trial_seed=seed + idx,
            )
            for idx in range(timed_runs)
        ]
        results.append(
            {
                "prompt_target": prompt_target,
                "trials": trials,
                "summary": _summarize_trials(trials),
            }
        )

    return {
        "surface": "AC2 scheduler-level canonical fast path",
        "model_path": str(CANONICAL_MODEL_PATH),
        "n": n,
        "max_tokens": max_tokens,
        "prefill_step_size": prefill_step_size,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-targets", type=int, nargs="+", default=(256, 2048))
    args = parser.parse_args()

    payload = run_probe(
        prompt_targets=tuple(args.prompt_targets),
        n=max(1, args.n),
        max_tokens=max(1, args.max_tokens),
        warmup_runs=max(0, args.warmup),
        timed_runs=max(1, args.runs),
        prefill_step_size=max(1, args.prefill_step),
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
