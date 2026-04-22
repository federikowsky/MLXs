"""Direct AC2 compare between MLXs BatchScheduler and mlx_lm BatchGenerator."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from transformers import AutoTokenizer

from benchmarks.mlxs_vs_mlx_lm.harness import CANONICAL_MODEL_PATH, _build_prompt_token_ids
from benchmarks.mlxs_vs_mlx_lm.memory import rss_bytes_self
from benchmarks.mlxs_vs_mlx_lm.scheduler_ac2_probe import run_probe as run_mlxs_probe
from benchmarks.mlxs_vs_mlx_lm.scheduler_ac2_utils import (
    compare_prompt_results,
    p95,
    summarize_probe_trials,
)


def _resolve_stop_tokens(tokenizer: Any) -> list[list[int]] | None:
    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        return [[int(eos_token_id)]] if eos_token_id is not None else None
    if isinstance(eos_token_ids, int):
        return [[int(eos_token_ids)]]
    return [[int(token_id)] for token_id in eos_token_ids]


def _run_mlx_lm_case(
    model: Any,
    tokenizer: Any,
    *,
    n: int,
    max_tokens: int,
    prompt_target: int,
    prefill_step_size: int,
    trial_seed: int,
) -> dict[str, Any]:
    mlx_lm_generate = importlib.import_module("mlx_lm.generate")
    prompt = list(_build_prompt_token_ids(tokenizer, prompt_target))
    generator = mlx_lm_generate.BatchGenerator(
        model,
        stop_tokens=_resolve_stop_tokens(tokenizer),
        completion_batch_size=n,
        prefill_batch_size=n,
        prefill_step_size=prefill_step_size,
    )

    outputs = {f"r{idx}": [] for idx in range(n)}
    first_token_at: dict[str, float] = {}
    completion_at: dict[str, float] = {}
    pending: set[int] = set()

    mx.random.seed(int(trial_seed) % (2**32))
    mx.reset_peak_memory()
    mx.clear_cache()
    try:
        t0 = time.perf_counter()
        uids = generator.insert([prompt for _ in range(n)], max_tokens=[max_tokens] * n)
        request_ids = {int(uid): f"r{idx}" for idx, uid in enumerate(uids)}
        pending = set(int(uid) for uid in uids)
        while pending:
            responses = generator.next_generated()
            now = time.perf_counter()
            for response in responses:
                request_id = request_ids[int(response.uid)]
                outputs[request_id].append(int(response.token))
                if request_id not in first_token_at:
                    first_token_at[request_id] = now - t0
                if response.finish_reason is not None:
                    completion_at[request_id] = now - t0
                    pending.discard(int(response.uid))
        wall = time.perf_counter() - t0
    finally:
        generator.close()

    generated_tokens = sum(len(tokens) for tokens in outputs.values())
    return {
        "prompt_target": prompt_target,
        "trial_seed": trial_seed,
        "requests_per_s": n / wall,
        "generated_tok_per_s": generated_tokens / wall,
        "p50_ttft_s": statistics.median(first_token_at.values()),
        "p95_completion_s": p95(list(completion_at.values())),
        "rss_bytes": rss_bytes_self(),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "outputs": outputs,
    }


def run_mlx_lm_probe(
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
    from mlx_lm import load

    tokenizer = AutoTokenizer.from_pretrained(
        str(CANONICAL_MODEL_PATH),
        trust_remote_code=trust_remote_code,
    )
    model, _ = load(
        str(CANONICAL_MODEL_PATH),
        tokenizer_config={"trust_remote_code": trust_remote_code},
        lazy=False,
    )

    results = []
    for prompt_target in prompt_targets:
        for warm_idx in range(warmup_runs):
            _run_mlx_lm_case(
                model,
                tokenizer,
                n=n,
                max_tokens=max_tokens,
                prompt_target=prompt_target,
                prefill_step_size=prefill_step_size,
                trial_seed=seed - 1000 - warm_idx,
            )
        trials = [
            _run_mlx_lm_case(
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
                "summary": summarize_probe_trials(trials),
            }
        )

    return {
        "surface": "AC2 scheduler-level canonical fast path",
        "backend": "mlx_lm.BatchGenerator",
        "model_path": str(CANONICAL_MODEL_PATH),
        "n": n,
        "max_tokens": max_tokens,
        "prefill_step_size": prefill_step_size,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "results": results,
    }


def run_compare(
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
    mlxs_payload = run_mlxs_probe(
        prompt_targets=prompt_targets,
        n=n,
        max_tokens=max_tokens,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        prefill_step_size=prefill_step_size,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )
    mlx_lm_payload = run_mlx_lm_probe(
        prompt_targets=prompt_targets,
        n=n,
        max_tokens=max_tokens,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        prefill_step_size=prefill_step_size,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )
    return {
        "surface": "AC2 direct canonical compare",
        "model_path": str(CANONICAL_MODEL_PATH),
        "n": n,
        "max_tokens": max_tokens,
        "prefill_step_size": prefill_step_size,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "prompt_targets": list(prompt_targets),
        "comparison_semantics": {
            "throughput_ratio": "higher is better for MLXs",
            "latency_ratio": "lower is better for MLXs",
        },
        "mlxs": mlxs_payload,
        "mlx_lm": mlx_lm_payload,
        "comparisons": compare_prompt_results(
            mlxs_payload["results"],
            mlx_lm_payload["results"],
        ),
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

    payload = run_compare(
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
