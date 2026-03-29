"""Orchestrate multi-model, multi-prompt MLXs vs mlx-lm benchmark sessions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer

from benchmarks.mlxs_vs_mlx_lm.backends import (
    GenerationMetrics,
    close_session,
    generate_mlx_lm_loaded,
    generate_mlxs_loaded,
    load_mlx_lm,
    load_mlxs,
)
from benchmarks.mlxs_vs_mlx_lm.discovery import (
    LocalModelInfo,
    discover_local_models,
    infer_hub_repo_id,
)
from benchmarks.mlxs_vs_mlx_lm.stats import TrialStats, summarize_trials
from benchmarks.mlxs_vs_mlx_lm.system_info import collect_fingerprint


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    warmup_runs: int
    timed_runs: int
    max_tokens: int
    seed: int
    prefill_step_size: int
    compile_decode: bool
    trust_remote_code: bool
    prompt_targets: tuple[int, ...]
    backends: frozenset[str]


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
    )


def _build_prompt_text(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    unit = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump.\n"
    )
    text = unit
    while len(tokenizer.encode(text)) < target_tokens + 16:
        text += unit
    ids = tokenizer.encode(text)
    ids = ids[:target_tokens]
    decoded = tokenizer.decode(ids)
    return decoded, len(ids)


def _finite(x: float) -> bool:
    return not math.isnan(x) and not math.isinf(x)


_FLOAT_METRIC_FIELDS = (
    "ttft_s",
    "prefill_effective_tok_per_s",
    "decode_tok_per_s",
    "end_to_end_wall_s",
    "end_to_end_tok_per_s",
)

_INT_METRIC_FIELDS = (
    "rss_bytes_after_generate",
    "mlx_peak_memory_bytes",
)


def _metrics_dict(m: GenerationMetrics) -> dict[str, Any]:
    d = asdict(m)
    return d


def _stats_dict(ts: TrialStats) -> dict[str, Any]:
    return asdict(ts)


def _summarize_field(trials: list[GenerationMetrics], field: str) -> TrialStats | None:
    if field in _FLOAT_METRIC_FIELDS:
        vals = []
        for t in trials:
            if t.error is not None:
                continue
            v = getattr(t, field)
            if isinstance(v, float) and _finite(v):
                vals.append(v)
        if not vals:
            return None
        return summarize_trials(vals)
    if field in _INT_METRIC_FIELDS:
        vals = []
        for t in trials:
            if t.error is not None:
                continue
            v = getattr(t, field)
            if v is not None:
                vals.append(float(v))
        if not vals:
            return None
        return summarize_trials(vals)
    return None


def _run_backend_session(
    backend: Literal["mlxs", "mlx_lm"],
    model_path: Path,
    prompt_text: str,
    cfg: HarnessConfig,
) -> tuple[dict[str, Any], list[GenerationMetrics]]:
    """Load once, warmup, timed trials, unload. Returns session header + trials."""
    if backend == "mlxs":
        session = load_mlxs(model_path, trust_remote_code=cfg.trust_remote_code)
    else:
        session = load_mlx_lm(model_path, trust_remote_code=cfg.trust_remote_code)

    header: dict[str, Any] = {
        "backend": backend,
        "load_wall_s": session.load_wall_s,
        "rss_bytes_before_load": session.rss_bytes_before_load,
        "rss_bytes_after_load": session.rss_bytes_after_load,
        "load_error": session.error,
    }

    trials: list[GenerationMetrics] = []
    if session.error:
        close_session(session)
        return header, trials

    warm = max(0, cfg.warmup_runs)
    warm_tokens = max(1, min(16, cfg.max_tokens))
    for _ in range(warm):
        if backend == "mlxs":
            generate_mlxs_loaded(
                session,
                prompt_text,
                max_tokens=warm_tokens,
                seed=cfg.seed,
                prefill_step_size=cfg.prefill_step_size,
                compile_decode=cfg.compile_decode,
            )
        else:
            generate_mlx_lm_loaded(
                session,
                prompt_text,
                max_tokens=warm_tokens,
                seed=cfg.seed,
                prefill_step_size=cfg.prefill_step_size,
            )

    for i in range(cfg.timed_runs):
        seed = cfg.seed + i
        if backend == "mlxs":
            m = generate_mlxs_loaded(
                session,
                prompt_text,
                max_tokens=cfg.max_tokens,
                seed=seed,
                prefill_step_size=cfg.prefill_step_size,
                compile_decode=cfg.compile_decode,
            )
        else:
            m = generate_mlx_lm_loaded(
                session,
                prompt_text,
                max_tokens=cfg.max_tokens,
                seed=seed,
                prefill_step_size=cfg.prefill_step_size,
            )
        trials.append(m)

    close_session(session)
    return header, trials


def run_harness(
    models: list[LocalModelInfo],
    cfg: HarnessConfig,
) -> dict[str, Any]:
    fingerprint = collect_fingerprint()
    results: list[dict[str, Any]] = []

    for mi in models:
        tok_path = str(mi.path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tok_path,
                trust_remote_code=cfg.trust_remote_code,
            )
        except Exception as exc:
            results.append(
                {
                    "model_hub_id": mi.hub_repo_id,
                    "model_path": str(mi.path),
                    "model_type": mi.model_type,
                    "skipped": True,
                    "skip_reason": f"tokenizer_load:{type(exc).__name__}:{exc}",
                }
            )
            continue

        for target in cfg.prompt_targets:
            prompt_text, enc_len = _build_prompt_text(tokenizer, target)
            row: dict[str, Any] = {
                "model_hub_id": mi.hub_repo_id,
                "model_path": str(mi.path),
                "model_type": mi.model_type,
                "weight_bytes": mi.weight_bytes,
                "prompt_target_tokens": target,
                "prompt_encoded_tokens_reference": enc_len,
                "skipped": False,
            }

            if "mlx_lm" in cfg.backends:
                mlx_head, mlx_trials = _run_backend_session("mlx_lm", mi.path, prompt_text, cfg)
                row["mlx_lm"] = {
                    "session": mlx_head,
                    "trials": [_metrics_dict(t) for t in mlx_trials],
                    "stats": {
                        f: _stats_dict(s)
                        for f in (*_FLOAT_METRIC_FIELDS, *_INT_METRIC_FIELDS)
                        if (s := _summarize_field(mlx_trials, f)) is not None
                    },
                }
            if "mlxs" in cfg.backends:
                xs_head, xs_trials = _run_backend_session("mlxs", mi.path, prompt_text, cfg)
                row["mlxs"] = {
                    "session": xs_head,
                    "trials": [_metrics_dict(t) for t in xs_trials],
                    "stats": {
                        f: _stats_dict(s)
                        for f in (*_FLOAT_METRIC_FIELDS, *_INT_METRIC_FIELDS)
                        if (s := _summarize_field(xs_trials, f)) is not None
                    },
                }

            if "mlx_lm" in cfg.backends and "mlxs" in cfg.backends:
                ratios: dict[str, float] = {}
                for key in (
                    "decode_tok_per_s",
                    "prefill_effective_tok_per_s",
                    "end_to_end_tok_per_s",
                ):
                    mlx_s = row["mlx_lm"]["stats"].get(key)
                    xs_s = row["mlxs"]["stats"].get(key)
                    if not mlx_s or not xs_s:
                        continue
                    denom = float(mlx_s["median"])
                    numer = float(xs_s["median"])
                    if denom > 0 and math.isfinite(denom) and math.isfinite(numer):
                        r = numer / denom
                        if math.isfinite(r):
                            ratios[f"mlxs_over_mlx_lm_{key}_median_ratio"] = r
                row["comparison"] = {"median_ratios": ratios}

            results.append(row)

    return {
        "fingerprint": fingerprint.to_json_dict(),
        "config": {
            "warmup_runs": cfg.warmup_runs,
            "timed_runs": cfg.timed_runs,
            "max_tokens": cfg.max_tokens,
            "base_seed": cfg.seed,
            "prefill_step_size": cfg.prefill_step_size,
            "compile_decode": cfg.compile_decode,
            "trust_remote_code": cfg.trust_remote_code,
            "prompt_targets": list(cfg.prompt_targets),
            "backends": sorted(cfg.backends),
        },
        "results": results,
    }


def resolve_models(
    *,
    hub_root: Path | None,
    max_models: int | None,
    explicit_paths: list[Path] | None,
) -> list[LocalModelInfo]:
    if explicit_paths:
        out: list[LocalModelInfo] = []
        for p in explicit_paths:
            info = local_model_from_path(p)
            if info is not None:
                out.append(info)
        return out
    return discover_local_models(hub_root, max_models=max_models)
