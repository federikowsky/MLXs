"""Orchestrate multi-model, multi-prompt MLXs vs mlx-lm benchmark sessions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer

from benchmarks.mlxs_vs_mlx_lm.backends import (
    GenerationMetrics,
    LoadSession,
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
from benchmarks.mlxs_vs_mlx_lm.scenarios import ScenarioSpec
from benchmarks.mlxs_vs_mlx_lm.stats import TrialStats, summarize_trials
from benchmarks.mlxs_vs_mlx_lm.system_info import collect_fingerprint


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    warmup_runs: int
    timed_runs: int
    seed: int
    prefill_step_size: int
    trust_remote_code: bool
    prompt_targets: tuple[int, ...]
    backends: frozenset[str]
    scenarios: tuple[ScenarioSpec, ...]


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
    "generated_tokens",
)


def _median_generated_tokens_comparable(mlx_median: float, xs_median: float) -> bool:
    """True when median output lengths are close enough for e2e tok/s ratio to be meaningful."""
    if not math.isfinite(mlx_median) or not math.isfinite(xs_median):
        return False
    a = float(mlx_median)
    b = float(xs_median)
    diff = abs(a - b)
    ref = max(abs(a), abs(b), 1.0)
    # Allow ≤2 tokens or ≤2% relative difference (whichever is looser in absolute terms).
    return diff <= max(2.0, 0.02 * ref)


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


def _session_header(session: LoadSession, backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "load_wall_s": session.load_wall_s,
        "rss_bytes_before_load": session.rss_bytes_before_load,
        "rss_bytes_after_load": session.rss_bytes_after_load,
        "load_error": session.error,
    }


def _trials_on_loaded_session(
    session: LoadSession,
    backend: Literal["mlxs", "mlx_lm"],
    prompt_text: str,
    cfg: HarnessConfig,
    spec: ScenarioSpec,
) -> list[GenerationMetrics]:
    """Warmup + timed runs; model must already be loaded (no load/unload)."""
    trials: list[GenerationMetrics] = []
    if session.error:
        return trials

    warm = max(0, cfg.warmup_runs)
    warm_tokens = max(1, min(16, spec.max_tokens))
    warm_spec = replace(spec, id=f"{spec.id}__warmup", max_tokens=warm_tokens)
    for _ in range(warm):
        if backend == "mlxs":
            generate_mlxs_loaded(
                session,
                prompt_text,
                spec=warm_spec,
                seed=cfg.seed,
                prefill_step_size=cfg.prefill_step_size,
            )
        else:
            generate_mlx_lm_loaded(
                session,
                prompt_text,
                spec=warm_spec,
                seed=cfg.seed,
                prefill_step_size=cfg.prefill_step_size,
            )

    for i in range(cfg.timed_runs):
        seed = cfg.seed + i
        if backend == "mlxs":
            m = generate_mlxs_loaded(
                session,
                prompt_text,
                spec=spec,
                seed=seed,
                prefill_step_size=cfg.prefill_step_size,
            )
        else:
            m = generate_mlx_lm_loaded(
                session,
                prompt_text,
                spec=spec,
                seed=seed,
                prefill_step_size=cfg.prefill_step_size,
            )
        trials.append(m)
    return trials


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

        prompt_by_target: dict[int, tuple[str, int]] = {}
        for target in cfg.prompt_targets:
            prompt_by_target[target] = _build_prompt_text(tokenizer, target)

        # (scenario_id, prompt_target) -> partial row without backend blocks
        keyed: dict[tuple[str, int], dict[str, Any]] = {}
        for spec in cfg.scenarios:
            for target in cfg.prompt_targets:
                _pt, enc_len = prompt_by_target[target]
                keyed[(spec.id, target)] = {
                    "scenario_id": spec.id,
                    "scenario": spec.to_json_dict(),
                    "model_hub_id": mi.hub_repo_id,
                    "model_path": str(mi.path),
                    "model_type": mi.model_type,
                    "weight_bytes": mi.weight_bytes,
                    "prompt_target_tokens": target,
                    "prompt_encoded_tokens_reference": enc_len,
                    "skipped": False,
                }

        backend_order: tuple[Literal["mlx_lm", "mlxs"], ...] = ("mlx_lm", "mlxs")
        for backend in backend_order:
            if backend not in cfg.backends:
                continue
            session = (
                load_mlxs(mi.path, trust_remote_code=cfg.trust_remote_code)
                if backend == "mlxs"
                else load_mlx_lm(mi.path, trust_remote_code=cfg.trust_remote_code)
            )
            header = _session_header(session, backend)
            try:
                if session.error:
                    for spec in cfg.scenarios:
                        for target in cfg.prompt_targets:
                            row = keyed[(spec.id, target)]
                            row[backend] = {
                                "session": header,
                                "trials": [],
                                "stats": {},
                            }
                    continue

                for spec in cfg.scenarios:
                    for target in cfg.prompt_targets:
                        prompt_text, _enc = prompt_by_target[target]
                        trials = _trials_on_loaded_session(
                            session, backend, prompt_text, cfg, spec
                        )
                        row = keyed[(spec.id, target)]
                        row[backend] = {
                            "session": header,
                            "trials": [_metrics_dict(t) for t in trials],
                            "stats": {
                                f: _stats_dict(s)
                                for f in (*_FLOAT_METRIC_FIELDS, *_INT_METRIC_FIELDS)
                                if (s := _summarize_field(trials, f)) is not None
                            },
                        }
            finally:
                close_session(session)

        for spec in cfg.scenarios:
            for target in cfg.prompt_targets:
                row = keyed[(spec.id, target)]
                if "mlx_lm" in cfg.backends and "mlxs" in cfg.backends:
                    ratios: dict[str, float] = {}
                    for key in (
                        "decode_tok_per_s",
                        "prefill_effective_tok_per_s",
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

                    gt_mlx = row["mlx_lm"]["stats"].get("generated_tokens")
                    gt_xs = row["mlxs"]["stats"].get("generated_tokens")
                    mlx_gt_med = gt_mlx["median"] if gt_mlx else None
                    xs_gt_med = gt_xs["median"] if gt_xs else None
                    e2e_key = "end_to_end_tok_per_s"
                    mlx_f = (
                        float(mlx_gt_med)
                        if isinstance(mlx_gt_med, (int, float))
                        else float("nan")
                    )
                    xs_f = (
                        float(xs_gt_med)
                        if isinstance(xs_gt_med, (int, float))
                        else float("nan")
                    )
                    e2e_comparable = (
                        math.isfinite(mlx_f)
                        and math.isfinite(xs_f)
                        and _median_generated_tokens_comparable(mlx_f, xs_f)
                    )
                    e2e_suppressed: str | None = None
                    if e2e_comparable:
                        mlx_s = row["mlx_lm"]["stats"].get(e2e_key)
                        xs_s = row["mlxs"]["stats"].get(e2e_key)
                        if mlx_s and xs_s:
                            denom = float(mlx_s["median"])
                            numer = float(xs_s["median"])
                            if denom > 0 and math.isfinite(denom) and math.isfinite(numer):
                                r = numer / denom
                                if math.isfinite(r):
                                    ratios[f"mlxs_over_mlx_lm_{e2e_key}_median_ratio"] = r
                    else:
                        if math.isfinite(mlx_f) and math.isfinite(xs_f):
                            e2e_suppressed = (
                                "end_to_end_tok_per_s_ratio_omitted: median "
                                "generated_tokens differ (not comparable; e.g. early EOS vs "
                                "max_tokens)."
                            )
                        else:
                            e2e_suppressed = (
                                "end_to_end_tok_per_s_ratio_omitted: missing median "
                                "generated_tokens for one or both backends."
                            )

                    row["comparison"] = {
                        "median_ratios": ratios,
                        "generated_tokens_median": {
                            "mlx_lm": mlx_gt_med,
                            "mlxs": xs_gt_med,
                        },
                        "end_to_end_tok_per_s_median_ratio_comparable": e2e_comparable,
                        "end_to_end_tok_per_s_median_ratio_suppressed_reason": e2e_suppressed,
                    }
                results.append(row)

    return {
        "fingerprint": fingerprint.to_json_dict(),
        "config": {
            "warmup_runs": cfg.warmup_runs,
            "timed_runs": cfg.timed_runs,
            "base_seed": cfg.seed,
            "prefill_step_size": cfg.prefill_step_size,
            "trust_remote_code": cfg.trust_remote_code,
            "prompt_targets": list(cfg.prompt_targets),
            "backends": sorted(cfg.backends),
            "scenarios": [s.to_json_dict() for s in cfg.scenarios],
            "load_once_per_backend": True,
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
