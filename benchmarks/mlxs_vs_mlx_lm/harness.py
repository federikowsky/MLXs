"""Orchestrate canonical Class A and exploratory benchmark sessions."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from benchmarks.mlxs_vs_mlx_lm.backends import (
    CANONICAL_BENCHMARK_CLASS,
    CANONICAL_CLEAR_CACHE_INTERVAL,
    GenerationMetrics,
    compiled_decode_eligible,
    close_session,
    generate_mlx_lm_loaded,
    generate_mlxs_loaded,
    load_mlx_lm,
    load_mlxs,
    runtime_profile,
)
from benchmarks.mlxs_vs_mlx_lm.discovery import (
    LocalModelInfo,
    discover_local_models,
    infer_hub_repo_id,
    infer_weight_format_class,
)
from benchmarks.mlxs_vs_mlx_lm.stats import TrialStats, summarize_trials
from benchmarks.mlxs_vs_mlx_lm.system_info import collect_fingerprint

CANONICAL_MODE = "canonical"
EXPLORATORY_MODE = "exploratory"
CANONICAL_MODEL_PATH = Path.home() / ".cache" / "huggingface" / "hub" / (
    "models--mlx-community--Llama-3.2-1B-Instruct-4bit"
) / "snapshots" / "08231374eeacb049a0eade7922910865b8fce912"
CANONICAL_PROMPT_TARGETS = (256, 2048)
CANONICAL_MAX_TOKENS = 128
CANONICAL_WARMUP_RUNS = 2
CANONICAL_TIMED_RUNS = 7


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    mode: Literal["canonical", "exploratory"]
    warmup_runs: int
    timed_runs: int
    max_tokens: int
    seed: int
    prefill_step_size: int
    trust_remote_code: bool
    prompt_targets: tuple[int, ...]
    backends: frozenset[str]
    clear_cache_interval: int = CANONICAL_CLEAR_CACHE_INTERVAL
    run_compiled_secondary: bool = True


def _load_reference_tokenizer(model_path: Path, *, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
    )


def local_model_from_path(path: Path) -> LocalModelInfo | None:
    """Treat ``path`` as a HF snapshot dir; return info or None if invalid."""
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    mt = raw.get("model_type")
    if not isinstance(mt, str):
        return None
    st_paths = list(path.glob("*.safetensors"))
    if not st_paths:
        return None
    arch = raw.get("architectures")
    arch_tuple = tuple(str(x) for x in arch) if isinstance(arch, list) else ()
    st = tuple(sorted(p.name for p in st_paths))
    wbytes = sum(p.stat().st_size for p in st_paths)
    hub = infer_hub_repo_id(path)
    return LocalModelInfo(
        path=path.resolve(),
        hub_repo_id=hub,
        model_type=mt,
        architectures=arch_tuple,
        safetensors_files=st,
        weight_bytes=wbytes,
        weight_format_class=infer_weight_format_class(raw, path),
    )


def _build_prompt_token_ids(tokenizer: Any, target_tokens: int) -> tuple[int, ...]:
    unit = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump.\n"
    )
    text = unit
    while len(tokenizer.encode(text)) < target_tokens + 16:
        text += unit
    ids = tokenizer.encode(text)
    return tuple(int(token_id) for token_id in ids[:target_tokens])


def _finite(x: float) -> bool:
    return not math.isnan(x) and not math.isinf(x)


_FLOAT_METRIC_FIELDS = (
    "prefill_wall_s",
    "ttft_s",
    "decode_wall_s",
    "end_to_end_wall_s",
    "prefill_tok_per_s",
    "decode_tok_per_s",
    "end_to_end_tok_per_s",
)

_INT_METRIC_FIELDS = (
    "prompt_tokens",
    "generated_tokens",
    "decode_tokens_measured",
    "rss_bytes_after_generate",
    "mlx_peak_memory_bytes",
)


def _metrics_dict(m: GenerationMetrics) -> dict[str, Any]:
    return asdict(m)


def _stats_dict(ts: TrialStats) -> dict[str, Any]:
    return asdict(ts)


def _summarize_field(trials: list[GenerationMetrics], field: str) -> TrialStats | None:
    if field in _FLOAT_METRIC_FIELDS:
        vals = []
        for trial in trials:
            if trial.error is not None:
                continue
            value = getattr(trial, field)
            if isinstance(value, float) and _finite(value):
                vals.append(value)
        if not vals:
            return None
        return summarize_trials(vals)
    if field in _INT_METRIC_FIELDS:
        vals = []
        for trial in trials:
            if trial.error is not None:
                continue
            value = getattr(trial, field)
            if value is not None:
                vals.append(float(value))
        if not vals:
            return None
        return summarize_trials(vals)
    return None


def _measurement_boundaries() -> dict[str, Any]:
    return {
        "prefill": {
            "start": "Immediately before the first runtime call that consumes shared prompt token ids.",
            "end": "Prompt ingestion complete and first-step logits or seed state ready, before first token availability.",
        },
        "ttft": {
            "start": "Same request-start boundary as prefill.",
            "end": "First generated token id becomes available to the harness after required eval/materialization.",
        },
        "decode": {
            "start": "Immediately after first-token availability.",
            "end": "Last-token availability for the fixed max-token target.",
            "metric": "decode_tok_per_s = (generated_tokens - 1) / decode_wall_s",
        },
        "end_to_end": {
            "start": "Same request-start boundary as prefill.",
            "end": "Last-token availability.",
            "role": "Secondary summary metric only for Class A.",
        },
    }


def _contamination_exclusions() -> dict[str, list[str]]:
    return {
        "layer2": [
            "no penalties/processors",
            "no rich sampling semantics",
            "no logprobs/top-logprobs reporting",
            "no stop-sequence semantics",
        ],
        "layer3": [
            "no batching",
            "no speculative decoding",
            "no prompt-cache reuse",
            "no scheduler/admission behavior",
        ],
        "layer4": [
            "no mlxs.generate.generate",
            "no server transport",
            "no SSE framing",
            "no CLI/product rendering",
            "no product-facing event objects",
        ],
        "diagnostics": [
            "no graph export",
            "no array printing",
            "no NumPy conversion",
            "no invasive profiling",
            "no extra scalar reads for instrumentation",
        ],
        "temporary_shims": [
            "no compatibility shim on the canonical benchmark path",
            "benchmark-local baseline helpers are labeled as benchmark-path behavior",
        ],
    }


def _benchmark_metadata(cfg: HarnessConfig) -> dict[str, Any]:
    return {
        "phase": "Phase 2",
        "benchmark_class": CANONICAL_BENCHMARK_CLASS,
        "benchmark_mode": cfg.mode,
        "warmup_rule": (
            f"{cfg.warmup_runs} fixed warmup runs per backend/prompt case; excluded from timed metrics."
        ),
        "aggregation_rule": "Median over timed runs with mean/stdev/min/max/p05/p95 retained.",
        "first_run_policy": "All timed trials are preserved for explicit first-run inspection.",
        "compile_policy": "MLXs eager primary canonical case; compiled MLXs separately labeled secondary variant.",
        "stream_policy": "Dedicated generation stream is explicit on both canonical paths.",
        "cache_policy": (
            f"Prefill clears cache per chunk; decode uses parameter {cfg.clear_cache_interval} "
            "with backend-specific runtime semantics recorded explicitly."
        ),
        "diagnostics_disabled": True,
        "graph_export_disabled": True,
    }


def _fairness_contract(cfg: HarnessConfig) -> dict[str, Any]:
    return {
        "same_model_policy": "Both sides load the same fixed model artifact for canonical sign-off.",
        "same_weight_format_policy": "Weight-format class is recorded per row and must match for fair comparison.",
        "same_tokenization_policy": (
            "Shared prompt token ids are built once outside the timed region and passed unchanged to both backends."
        ),
        "same_prompt_decode_targets": {
            "prompt_targets": list(cfg.prompt_targets),
            "decode_target_tokens": cfg.max_tokens,
            "finish_policy": "fixed_max_tokens_only",
        },
        "runtime_profiles": {
            "mlx_lm": runtime_profile("mlx_lm", clear_cache_interval=cfg.clear_cache_interval),
            "mlxs_eager": runtime_profile(
                "mlxs_eager",
                clear_cache_interval=cfg.clear_cache_interval,
            ),
            "mlxs_compiled": runtime_profile(
                "mlxs_compiled",
                clear_cache_interval=cfg.clear_cache_interval,
            ),
        },
    }


def _run_backend_session(
    backend_variant: Literal["mlx_lm", "mlxs_eager", "mlxs_compiled"],
    model_path: Path,
    prompt_token_ids: tuple[int, ...],
    cfg: HarnessConfig,
) -> tuple[dict[str, Any], list[GenerationMetrics]]:
    """Load once, warm up, run timed trials, then unload."""
    if backend_variant == "mlx_lm":
        session = load_mlx_lm(model_path, trust_remote_code=cfg.trust_remote_code)
    else:
        session = load_mlxs(model_path, trust_remote_code=cfg.trust_remote_code)
    compile_enabled = backend_variant == "mlxs_compiled" and compiled_decode_eligible(
        prompt_token_count=len(prompt_token_ids),
    )

    header: dict[str, Any] = {
        "backend": "mlx_lm" if backend_variant == "mlx_lm" else "mlxs",
        "variant": backend_variant,
        "load_wall_s": session.load_wall_s,
        "rss_bytes_before_load": session.rss_bytes_before_load,
        "rss_bytes_after_load": session.rss_bytes_after_load,
        "load_error": session.error,
        "compile_enabled": compile_enabled,
        "warmup_runs": cfg.warmup_runs,
        "timed_runs": cfg.timed_runs,
        "warmup_generated_tokens": max(1, min(16, cfg.max_tokens)),
        "runtime_profile": runtime_profile(
            backend_variant,
            clear_cache_interval=cfg.clear_cache_interval,
        ),
    }

    trials: list[GenerationMetrics] = []
    if session.error:
        close_session(session)
        return header, trials

    warm_tokens = header["warmup_generated_tokens"]
    warmup_wall_start = time.perf_counter()
    for _ in range(max(0, cfg.warmup_runs)):
        if backend_variant == "mlx_lm":
            generate_mlx_lm_loaded(
                session,
                prompt_token_ids,
                max_tokens=warm_tokens,
                seed=cfg.seed,
                prefill_step_size=cfg.prefill_step_size,
                clear_cache_interval=cfg.clear_cache_interval,
            )
        else:
            generate_mlxs_loaded(
                session,
                prompt_token_ids,
                max_tokens=warm_tokens,
                seed=cfg.seed,
                prefill_step_size=cfg.prefill_step_size,
                clear_cache_interval=cfg.clear_cache_interval,
                compile_decode=backend_variant == "mlxs_compiled",
            )
    warmup_wall_s = time.perf_counter() - warmup_wall_start
    header["warmup_wall_s"] = warmup_wall_s
    header["compile_warmup_wall_s"] = (
        warmup_wall_s if backend_variant == "mlxs_compiled" else None
    )

    for index in range(cfg.timed_runs):
        seed = cfg.seed + index
        if backend_variant == "mlx_lm":
            metrics = generate_mlx_lm_loaded(
                session,
                prompt_token_ids,
                max_tokens=cfg.max_tokens,
                seed=seed,
                prefill_step_size=cfg.prefill_step_size,
                clear_cache_interval=cfg.clear_cache_interval,
            )
        else:
            metrics = generate_mlxs_loaded(
                session,
                prompt_token_ids,
                max_tokens=cfg.max_tokens,
                seed=seed,
                prefill_step_size=cfg.prefill_step_size,
                clear_cache_interval=cfg.clear_cache_interval,
                compile_decode=backend_variant == "mlxs_compiled",
            )
        trials.append(metrics)

    close_session(session)
    return header, trials


def run_harness(
    models: list[LocalModelInfo],
    cfg: HarnessConfig,
) -> dict[str, Any]:
    fingerprint = collect_fingerprint()
    results: list[dict[str, Any]] = []

    for model_info in models:
        try:
            tokenizer = _load_reference_tokenizer(
                model_info.path,
                trust_remote_code=cfg.trust_remote_code,
            )
        except Exception as exc:
            results.append(
                {
                    "benchmark_class": CANONICAL_BENCHMARK_CLASS,
                    "benchmark_mode": cfg.mode,
                    "model_hub_id": model_info.hub_repo_id,
                    "model_path": str(model_info.path),
                    "model_type": model_info.model_type,
                    "weight_format_class": model_info.weight_format_class,
                    "skipped": True,
                    "skip_reason": f"tokenizer_load:{type(exc).__name__}:{exc}",
                }
            )
            continue

        for target in cfg.prompt_targets:
            prompt_token_ids = _build_prompt_token_ids(tokenizer, target)
            row: dict[str, Any] = {
                "benchmark_class": CANONICAL_BENCHMARK_CLASS,
                "benchmark_mode": cfg.mode,
                "model_hub_id": model_info.hub_repo_id,
                "model_path": str(model_info.path),
                "model_type": model_info.model_type,
                "architectures": list(model_info.architectures),
                "weight_bytes": model_info.weight_bytes,
                "weight_format_class": model_info.weight_format_class,
                "prompt_target_tokens": target,
                "prompt_token_count": len(prompt_token_ids),
                "prompt_input_policy": "shared_pre_tokenized_ids",
                "decode_target_tokens": cfg.max_tokens,
                "finish_condition_policy": "fixed_max_tokens_only",
                "skipped": False,
            }

            if "mlx_lm" in cfg.backends:
                mlx_head, mlx_trials = _run_backend_session(
                    "mlx_lm",
                    model_info.path,
                    prompt_token_ids,
                    cfg,
                )
                row["mlx_lm"] = {
                    "session": mlx_head,
                    "trials": [_metrics_dict(trial) for trial in mlx_trials],
                    "stats": {
                        field: _stats_dict(stats)
                        for field in (*_FLOAT_METRIC_FIELDS, *_INT_METRIC_FIELDS)
                        if (stats := _summarize_field(mlx_trials, field)) is not None
                    },
                }

            if "mlxs" in cfg.backends:
                eager_head, eager_trials = _run_backend_session(
                    "mlxs_eager",
                    model_info.path,
                    prompt_token_ids,
                    cfg,
                )
                row["mlxs_eager"] = {
                    "session": eager_head,
                    "trials": [_metrics_dict(trial) for trial in eager_trials],
                    "stats": {
                        field: _stats_dict(stats)
                        for field in (*_FLOAT_METRIC_FIELDS, *_INT_METRIC_FIELDS)
                        if (stats := _summarize_field(eager_trials, field)) is not None
                    },
                }

                if cfg.run_compiled_secondary:
                    compiled_head, compiled_trials = _run_backend_session(
                        "mlxs_compiled",
                        model_info.path,
                        prompt_token_ids,
                        cfg,
                    )
                    row["mlxs_compiled"] = {
                        "session": compiled_head,
                        "trials": [_metrics_dict(trial) for trial in compiled_trials],
                        "stats": {
                            field: _stats_dict(stats)
                            for field in (*_FLOAT_METRIC_FIELDS, *_INT_METRIC_FIELDS)
                            if (stats := _summarize_field(compiled_trials, field)) is not None
                        },
                    }

            comparisons: dict[str, float] = {}
            if "mlx_lm" in row and "mlxs_eager" in row:
                for key in ("decode_tok_per_s", "prefill_tok_per_s", "end_to_end_tok_per_s"):
                    baseline = row["mlx_lm"]["stats"].get(key)
                    eager = row["mlxs_eager"]["stats"].get(key)
                    if baseline and eager:
                        denom = float(baseline["median"])
                        numer = float(eager["median"])
                        if denom > 0 and math.isfinite(denom) and math.isfinite(numer):
                            comparisons[f"mlxs_eager_over_mlx_lm_{key}_median_ratio"] = numer / denom

            if "mlx_lm" in row and "mlxs_compiled" in row:
                for key in ("decode_tok_per_s", "prefill_tok_per_s", "end_to_end_tok_per_s"):
                    baseline = row["mlx_lm"]["stats"].get(key)
                    compiled = row["mlxs_compiled"]["stats"].get(key)
                    if baseline and compiled:
                        denom = float(baseline["median"])
                        numer = float(compiled["median"])
                        if denom > 0 and math.isfinite(denom) and math.isfinite(numer):
                            comparisons[
                                f"mlxs_compiled_over_mlx_lm_{key}_median_ratio"
                            ] = numer / denom

            if comparisons:
                row["comparison"] = {"median_ratios": comparisons}

            results.append(row)

    return {
        "benchmark": _benchmark_metadata(cfg),
        "fingerprint": fingerprint.to_json_dict(),
        "config": {
            "mode": cfg.mode,
            "warmup_runs": cfg.warmup_runs,
            "timed_runs": cfg.timed_runs,
            "max_tokens": cfg.max_tokens,
            "base_seed": cfg.seed,
            "prefill_step_size": cfg.prefill_step_size,
            "clear_cache_interval": cfg.clear_cache_interval,
            "run_compiled_secondary": cfg.run_compiled_secondary,
            "trust_remote_code": cfg.trust_remote_code,
            "prompt_targets": list(cfg.prompt_targets),
            "backends": sorted(cfg.backends),
            "canonical_model_path": str(CANONICAL_MODEL_PATH),
        },
        "measurement_boundaries": _measurement_boundaries(),
        "fairness_contract": _fairness_contract(cfg),
        "contamination_exclusions": _contamination_exclusions(),
        "results": results,
    }


def resolve_models(
    *,
    hub_root: Path | None,
    max_models: int | None,
    explicit_paths: list[Path] | None,
    mode: Literal["canonical", "exploratory"],
) -> list[LocalModelInfo]:
    if mode == CANONICAL_MODE:
        model = local_model_from_path(CANONICAL_MODEL_PATH)
        return [model] if model is not None else []
    if explicit_paths:
        out: list[LocalModelInfo] = []
        for path in explicit_paths:
            info = local_model_from_path(path)
            if info is not None:
                out.append(info)
        return out
    return discover_local_models(hub_root, max_models=max_models)
