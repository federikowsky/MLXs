"""R30 AC2 lower-boundary probe on the staggered late-admission surface."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from benchmarks.mlxs_vs_mlx_lm.harness import CANONICAL_MODEL_PATH, _build_prompt_token_ids
from benchmarks.mlxs_vs_mlx_lm.memory import rss_bytes_self
from mlxs._types import GenerateOptions, ModelMode
from mlxs.batch.scheduler import BatchScheduler
from mlxs.load.loader import load_model


def _require_mx() -> Any:
    import mlx.core as mx

    return mx


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _model_kind(model_path: str) -> str:
    if "Llama-3.2" in model_path:
        return "llama"
    if "Qwen2.5" in model_path:
        return "qwen2"
    raise ValueError(f"Unsupported R30 substrate probe model: {model_path}")


def _compare_tokens(left: list[int], right: list[int], *, prefix: int = 16) -> dict[str, Any]:
    mismatch = None
    for idx, (lhs, rhs) in enumerate(zip(left, right, strict=False)):
        if lhs != rhs:
            mismatch = idx
            break
    if mismatch is None and len(left) != len(right):
        mismatch = min(len(left), len(right))
    return {
        "exact": left == right,
        "same_count": len(left) == len(right),
        "prefix_match": left[:prefix] == right[:prefix],
        "prefix_length": prefix,
        "first_mismatch_index": mismatch,
        "left_prefix": left[:prefix],
        "right_prefix": right[:prefix],
    }


def _resolve_stop_tokens(tokenizer: Any) -> list[list[int]] | None:
    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        return [[int(eos_token_id)]] if eos_token_id is not None else None
    if isinstance(eos_token_ids, int):
        return [[int(eos_token_ids)]]
    return [[int(token_id)] for token_id in eos_token_ids]


def _group_summary(
    first_token_at: dict[str, float],
    completion_at: dict[str, float],
    first_token_step: dict[str, int],
    completion_step: dict[str, int],
    request_ids: list[str],
) -> dict[str, float]:
    return {
        "ttft_s": _median([first_token_at[rid] for rid in request_ids]),
        "completion_s": _median([completion_at[rid] for rid in request_ids]),
        "first_token_step": _median([float(first_token_step[rid]) for rid in request_ids]),
        "completion_step": _median([float(completion_step[rid]) for rid in request_ids]),
    }


class _Recorder:
    def __init__(self) -> None:
        self.phase = "idle"
        self.components: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def add(self, key: str, dt: float) -> None:
        self.components[self.phase][key] += dt

    def bump(self, key: str, delta: int = 1) -> None:
        self.counts[self.phase][key] += delta

    def snapshot_components(self) -> dict[str, dict[str, float]]:
        return {
            phase: {key: float(value) for key, value in values.items()}
            for phase, values in self.components.items()
        }

    def snapshot_counts(self) -> dict[str, dict[str, int]]:
        return {
            phase: {key: int(value) for key, value in values.items()}
            for phase, values in self.counts.items()
        }


@contextmanager
def _patched_mx_runtime(recorder: _Recorder):
    mx = _require_mx()
    orig_eval = mx.eval
    orig_async_eval = mx.async_eval
    orig_clear_cache = mx.clear_cache

    def wrapped_eval(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_eval(*args, **kwargs)
        recorder.add("mx_eval_s", time.perf_counter() - t0)
        recorder.bump("mx_eval_calls")
        return out

    def wrapped_async_eval(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_async_eval(*args, **kwargs)
        recorder.add("mx_async_eval_s", time.perf_counter() - t0)
        recorder.bump("mx_async_eval_calls")
        return out

    def wrapped_clear_cache(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_clear_cache(*args, **kwargs)
        recorder.add("clear_cache_s", time.perf_counter() - t0)
        recorder.bump("clear_cache_calls")
        return out

    mx.eval = wrapped_eval
    mx.async_eval = wrapped_async_eval
    mx.clear_cache = wrapped_clear_cache
    try:
        yield
    finally:
        mx.eval = orig_eval
        mx.async_eval = orig_async_eval
        mx.clear_cache = orig_clear_cache


@contextmanager
def _patch_phase_method(target: Any, name: str, recorder: _Recorder, phase: str):
    original = getattr(target, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        prev_phase = recorder.phase
        recorder.phase = phase
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            recorder.add("wall_s", time.perf_counter() - t0)
            recorder.bump("calls")
            recorder.phase = prev_phase

    setattr(target, name, wrapped)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextmanager
def _patched_mlxs_substrate(recorder: _Recorder, model_kind: str):
    import mlxs.cache.kv as mlxs_kv

    model_module_name = {
        "llama": "mlxs.models.llama",
        "qwen2": "mlxs.models.qwen",
    }[model_kind]
    model_module = importlib.import_module(model_module_name)

    orig_mask = model_module.create_attention_mask
    orig_attention = model_module.scaled_dot_product_attention
    orig_update = mlxs_kv.KVCache.update_and_fetch

    def wrapped_mask(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_mask(*args, **kwargs)
        recorder.add("mask_s", time.perf_counter() - t0)
        recorder.bump("mask_calls")
        return out

    def wrapped_attention(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_attention(*args, **kwargs)
        recorder.add("attention_s", time.perf_counter() - t0)
        recorder.bump("attention_calls")
        return out

    def wrapped_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_update(self, *args, **kwargs)
        recorder.add("cache_update_s", time.perf_counter() - t0)
        recorder.bump("cache_update_calls")
        return out

    model_module.create_attention_mask = wrapped_mask
    model_module.scaled_dot_product_attention = wrapped_attention
    mlxs_kv.KVCache.update_and_fetch = wrapped_update
    try:
        yield
    finally:
        model_module.create_attention_mask = orig_mask
        model_module.scaled_dot_product_attention = orig_attention
        mlxs_kv.KVCache.update_and_fetch = orig_update


@contextmanager
def _patched_mlx_lm_substrate(recorder: _Recorder, model_kind: str):
    mlx_lm_cache = importlib.import_module("mlx_lm.models.cache")
    model_module_name = {
        "llama": "mlx_lm.models.llama",
        "qwen2": "mlx_lm.models.qwen2",
    }[model_kind]
    model_module = importlib.import_module(model_module_name)

    orig_mask = model_module.create_attention_mask
    orig_attention = model_module.scaled_dot_product_attention
    orig_update = mlx_lm_cache.KVCache.update_and_fetch

    def wrapped_mask(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_mask(*args, **kwargs)
        recorder.add("mask_s", time.perf_counter() - t0)
        recorder.bump("mask_calls")
        return out

    def wrapped_attention(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_attention(*args, **kwargs)
        recorder.add("attention_s", time.perf_counter() - t0)
        recorder.bump("attention_calls")
        return out

    def wrapped_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = orig_update(self, *args, **kwargs)
        recorder.add("cache_update_s", time.perf_counter() - t0)
        recorder.bump("cache_update_calls")
        return out

    model_module.create_attention_mask = wrapped_mask
    model_module.scaled_dot_product_attention = wrapped_attention
    mlx_lm_cache.KVCache.update_and_fetch = wrapped_update
    try:
        yield
    finally:
        model_module.create_attention_mask = orig_mask
        model_module.scaled_dot_product_attention = orig_attention
        mlx_lm_cache.KVCache.update_and_fetch = orig_update


def _run_mlxs_case(
    model: Any,
    tokenizer: Any,
    *,
    prompt_target: int,
    initial_requests: int,
    late_requests: int,
    late_after_steps: int,
    max_tokens: int,
    prefill_step_size: int,
    trial_seed: int,
    model_kind: str,
) -> dict[str, Any]:
    from mlxs.batch.fast_batch import _SharedFastBatch

    prompt = _build_prompt_token_ids(tokenizer, prompt_target)
    total_requests = initial_requests + late_requests
    scheduler = BatchScheduler(
        prefill_batch_size=total_requests,
        completion_batch_size=total_requests,
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

    initial_ids = [f"r{idx}" for idx in range(initial_requests)]
    late_ids = [f"r{idx}" for idx in range(initial_requests, total_requests)]
    outputs = {f"r{idx}": [] for idx in range(total_requests)}
    first_token_at: dict[str, float] = {}
    completion_at: dict[str, float] = {}
    first_token_step: dict[str, int] = {}
    completion_step: dict[str, int] = {}
    pending = set(initial_ids)

    for request_id in initial_ids:
        scheduler.add(request_id, model, tokenizer, prompt, options)

    recorder = _Recorder()
    late_added = False
    steps = 0
    mx = _require_mx()
    mx.random.seed(int(trial_seed) % (2**32))
    mx.reset_peak_memory()
    mx.clear_cache()
    t0 = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(_patched_mx_runtime(recorder))
        stack.enter_context(_patched_mlxs_substrate(recorder, model_kind))
        stack.enter_context(
            _patch_phase_method(
                BatchScheduler,
                "step",
                recorder,
                "scheduler_step",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                BatchScheduler,
                "_prefill_pending",
                recorder,
                "prefill_pending",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                BatchScheduler,
                "_prefill_one",
                recorder,
                "prefill_one",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                BatchScheduler,
                "_prefill_cohort",
                recorder,
                "prefill_cohort",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                BatchScheduler,
                "_activate_fast_batch",
                recorder,
                "activate_fast_batch",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                BatchScheduler,
                "_decode_active",
                recorder,
                "decode_active",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                BatchScheduler,
                "_merge_group_cache",
                recorder,
                "merge_group_cache",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                BatchScheduler,
                "_scatter_group_cache",
                recorder,
                "scatter_group_cache",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                _SharedFastBatch,
                "step",
                recorder,
                "fast_batch_step",
            )
        )
        while pending:
            step_results = scheduler.step()
            steps += 1
            now = time.perf_counter()
            for request_id, events in step_results.items():
                outputs[request_id].extend(int(event.token_id) for event in events)
                if request_id not in first_token_at and events:
                    first_token_at[request_id] = now - t0
                    first_token_step[request_id] = steps

            if not late_added and steps >= late_after_steps:
                for request_id in late_ids:
                    scheduler.add(request_id, model, tokenizer, prompt, options)
                    pending.add(request_id)
                late_added = True

            for request_id, _events, _final_cache in scheduler.drain():
                completion_at[request_id] = now - t0
                completion_step[request_id] = steps
                pending.discard(request_id)

    return {
        "backend": "mlxs.BatchScheduler",
        "prompt_target": prompt_target,
        "trial_seed": trial_seed,
        "total_steps": steps,
        "initial_ids": initial_ids,
        "late_ids": late_ids,
        "initial_summary": _group_summary(
            first_token_at,
            completion_at,
            first_token_step,
            completion_step,
            initial_ids,
        ),
        "late_summary": _group_summary(
            first_token_at,
            completion_at,
            first_token_step,
            completion_step,
            late_ids,
        ),
        "components": recorder.snapshot_components(),
        "counts": recorder.snapshot_counts(),
        "rss_bytes": rss_bytes_self(),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "outputs": outputs,
    }


def _run_mlx_lm_case(
    model: Any,
    tokenizer: Any,
    *,
    prompt_target: int,
    initial_requests: int,
    late_requests: int,
    late_after_steps: int,
    max_tokens: int,
    prefill_step_size: int,
    trial_seed: int,
    model_kind: str,
) -> dict[str, Any]:
    mlx_lm_generate = importlib.import_module("mlx_lm.generate")
    prompt = list(_build_prompt_token_ids(tokenizer, prompt_target))
    total_requests = initial_requests + late_requests
    generator = mlx_lm_generate.BatchGenerator(
        model,
        stop_tokens=_resolve_stop_tokens(tokenizer),
        completion_batch_size=total_requests,
        prefill_batch_size=total_requests,
        prefill_step_size=prefill_step_size,
    )

    initial_ids = [f"r{idx}" for idx in range(initial_requests)]
    late_ids = [f"r{idx}" for idx in range(initial_requests, total_requests)]
    outputs = {f"r{idx}": [] for idx in range(total_requests)}
    first_token_at: dict[str, float] = {}
    completion_at: dict[str, float] = {}
    first_token_step: dict[str, int] = {}
    completion_step: dict[str, int] = {}
    pending: set[int] = set()
    recorder = _Recorder()
    mx = _require_mx()
    mx.random.seed(int(trial_seed) % (2**32))
    mx.reset_peak_memory()
    mx.clear_cache()
    with ExitStack() as stack:
        stack.enter_context(_patched_mx_runtime(recorder))
        stack.enter_context(_patched_mlx_lm_substrate(recorder, model_kind))
        stack.enter_context(
            _patch_phase_method(
                mlx_lm_generate.BatchGenerator,
                "_next",
                recorder,
                "batch_generator_next",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                mlx_lm_generate.PromptProcessingBatch,
                "prompt",
                recorder,
                "prompt_batch_prompt",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                mlx_lm_generate.PromptProcessingBatch,
                "generate",
                recorder,
                "prompt_batch_generate",
            )
        )
        stack.enter_context(
            _patch_phase_method(
                mlx_lm_generate.GenerationBatch,
                "_step",
                recorder,
                "generation_batch_step",
            )
        )
        try:
            uids = generator.insert(
                [prompt for _ in range(initial_requests)],
                max_tokens=[max_tokens] * initial_requests,
            )
            request_ids = {int(uid): f"r{idx}" for idx, uid in enumerate(uids)}
            pending.update(int(uid) for uid in uids)

            late_added = False
            steps = 0
            t0 = time.perf_counter()
            while pending:
                responses = generator.next_generated()
                steps += 1
                now = time.perf_counter()
                for response in responses:
                    request_id = request_ids[int(response.uid)]
                    outputs[request_id].append(int(response.token))
                    if request_id not in first_token_at:
                        first_token_at[request_id] = now - t0
                        first_token_step[request_id] = steps
                    if response.finish_reason is not None:
                        completion_at[request_id] = now - t0
                        completion_step[request_id] = steps
                        pending.discard(int(response.uid))

                if not late_added and steps >= late_after_steps:
                    start = initial_requests
                    late_uids = generator.insert(
                        [prompt for _ in range(late_requests)],
                        max_tokens=[max_tokens] * late_requests,
                    )
                    for offset, uid in enumerate(late_uids, start=start):
                        request_ids[int(uid)] = f"r{offset}"
                        pending.add(int(uid))
                    late_added = True
        finally:
            generator.close()

    return {
        "backend": "mlx_lm.BatchGenerator",
        "prompt_target": prompt_target,
        "trial_seed": trial_seed,
        "total_steps": steps,
        "initial_ids": initial_ids,
        "late_ids": late_ids,
        "initial_summary": _group_summary(
            first_token_at,
            completion_at,
            first_token_step,
            completion_step,
            initial_ids,
        ),
        "late_summary": _group_summary(
            first_token_at,
            completion_at,
            first_token_step,
            completion_step,
            late_ids,
        ),
        "components": recorder.snapshot_components(),
        "counts": recorder.snapshot_counts(),
        "rss_bytes": rss_bytes_self(),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "outputs": outputs,
    }


def run_probe(
    *,
    prompt_target: int,
    initial_requests: int,
    late_requests: int,
    late_after_steps: int,
    max_tokens: int,
    prefill_step_size: int,
    trust_remote_code: bool,
    seed: int,
) -> dict[str, Any]:
    model_path = str(CANONICAL_MODEL_PATH)
    model_kind = _model_kind(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    mlxs_model = load_model(
        CANONICAL_MODEL_PATH,
        lazy=False,
        model_mode=ModelMode.AUTO,
    )
    from mlx_lm import load as load_mlx_lm

    mlx_lm_model, _ = load_mlx_lm(
        model_path,
        tokenizer_config={"trust_remote_code": trust_remote_code},
        lazy=False,
    )

    mlxs = _run_mlxs_case(
        mlxs_model,
        tokenizer,
        prompt_target=prompt_target,
        initial_requests=initial_requests,
        late_requests=late_requests,
        late_after_steps=late_after_steps,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        trial_seed=seed,
        model_kind=model_kind,
    )
    mlx_lm = _run_mlx_lm_case(
        mlx_lm_model,
        tokenizer,
        prompt_target=prompt_target,
        initial_requests=initial_requests,
        late_requests=late_requests,
        late_after_steps=late_after_steps,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        trial_seed=seed,
        model_kind=model_kind,
    )
    return {
        "surface": "R30 staggered AC2 lower-boundary probe",
        "model_path": model_path,
        "model_kind": model_kind,
        "prompt_target": prompt_target,
        "initial_requests": initial_requests,
        "late_requests": late_requests,
        "late_after_steps": late_after_steps,
        "max_tokens": max_tokens,
        "prefill_step_size": prefill_step_size,
        "mlxs": mlxs,
        "mlx_lm": mlx_lm,
        "exact_output_parity": mlxs["outputs"] == mlx_lm["outputs"],
        "token_parity": {
            request_id: _compare_tokens(mlxs["outputs"][request_id], mlx_lm["outputs"][request_id])
            for request_id in mlxs["outputs"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-target", type=int, default=256)
    parser.add_argument("--initial-requests", type=int, default=2)
    parser.add_argument("--late-requests", type=int, default=1)
    parser.add_argument("--late-after-steps", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    payload = run_probe(
        prompt_target=max(2, args.prompt_target),
        initial_requests=max(1, args.initial_requests),
        late_requests=max(1, args.late_requests),
        late_after_steps=max(1, args.late_after_steps),
        max_tokens=max(1, args.max_tokens),
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
