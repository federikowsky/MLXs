"""Load + timed generation for MLXs and mlx-lm (greedy, no Adaptive KV).

Session pattern: load once per (backend, model path), then many generate calls.
This matches standard prefill/decode throughput methodology (load not in hot loop).
"""

from __future__ import annotations

import contextlib
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from benchmarks.mlxs_vs_mlx_lm.memory import rss_bytes_self


@dataclass(slots=True)
class GenerationMetrics:
    """One greedy generation after the model is already in memory."""

    backend: str
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float
    prefill_effective_tok_per_s: float
    decode_wall_s: float
    decode_tok_per_s: float
    end_to_end_wall_s: float
    end_to_end_tok_per_s: float
    rss_bytes_after_generate: int
    mlx_peak_memory_bytes: int | None
    error: str | None = None


@dataclass(slots=True)
class LoadSession:
    """Model resident in RAM; use for warmup + timed trials."""

    backend: str
    model: Any
    tokenizer: Any
    load_wall_s: float
    rss_bytes_before_load: int
    rss_bytes_after_load: int
    error: str | None = None


def _reset_mlx_memory_stats() -> None:
    with contextlib.suppress(Exception):
        mx.reset_peak_memory()


def _mlx_peak_bytes() -> int | None:
    try:
        return int(mx.get_peak_memory())
    except Exception:
        return None


def _unload_best_effort(model: Any, tokenizer: Any) -> None:
    del model
    del tokenizer
    gc.collect()
    with contextlib.suppress(Exception):
        mx.clear_cache()
    _reset_mlx_memory_stats()


def load_mlxs(
    model_path: Path,
    *,
    trust_remote_code: bool,
) -> LoadSession:
    from mlxs._types import ModelMode
    from mlxs.load.loader import load_model, load_tokenizer

    rss0 = rss_bytes_self()
    _reset_mlx_memory_stats()
    t0 = time.perf_counter()
    try:
        model = load_model(
            model_path,
            lazy=False,
            model_mode=ModelMode.AUTO,
        )
        tokenizer = load_tokenizer(model_path, trust_remote_code=trust_remote_code)
    except Exception as exc:
        return LoadSession(
            backend="mlxs",
            model=None,
            tokenizer=None,
            load_wall_s=time.perf_counter() - t0,
            rss_bytes_before_load=rss0,
            rss_bytes_after_load=rss_bytes_self(),
            error=f"{type(exc).__name__}: {exc}",
        )
    return LoadSession(
        backend="mlxs",
        model=model,
        tokenizer=tokenizer,
        load_wall_s=time.perf_counter() - t0,
        rss_bytes_before_load=rss0,
        rss_bytes_after_load=rss_bytes_self(),
        error=None,
    )


def generate_mlxs_loaded(
    session: LoadSession,
    prompt_text: str,
    *,
    max_tokens: int,
    seed: int,
    prefill_step_size: int,
    compile_decode: bool,
) -> GenerationMetrics:
    from mlxs._types import GenerateOptions
    from mlxs.generate import generate

    if session.error or session.model is None:
        return GenerationMetrics(
            backend="mlxs",
            prompt_tokens=0,
            generated_tokens=0,
            ttft_s=float("nan"),
            prefill_effective_tok_per_s=float("nan"),
            decode_wall_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_wall_s=float("nan"),
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=None,
            error=session.error or "no_model",
        )

    _reset_mlx_memory_stats()
    model = session.model
    tokenizer = session.tokenizer

    opts = GenerateOptions(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=seed,
    )
    mx.random.seed(seed)

    n_gen = 0
    ttft: float | None = None
    t_first: float | None = None
    t_end: float | None = None
    prompt_tokens = 0

    t_e2e0 = time.perf_counter()
    try:
        gen = generate(
            model,
            tokenizer,
            prompt_text,
            opts,
            prefill_step_size=prefill_step_size,
            compile_decode=compile_decode,
            clear_cache_interval=0,
        )
        for ev in gen:
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t_e2e0
                t_first = now
            prompt_tokens = ev.prompt_tokens
            n_gen = ev.generation_tokens
            if ev.finish_reason is not None:
                t_end = now
                break
    except Exception as exc:
        return GenerationMetrics(
            backend="mlxs",
            prompt_tokens=prompt_tokens,
            generated_tokens=n_gen,
            ttft_s=float("nan"),
            prefill_effective_tok_per_s=float("nan"),
            decode_wall_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_wall_s=time.perf_counter() - t_e2e0,
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=_mlx_peak_bytes(),
            error=f"{type(exc).__name__}: {exc}",
        )

    if t_end is None:
        t_end = time.perf_counter()
    if ttft is None or t_first is None:
        ttft_v = float("nan")
        decode_wall = float("nan")
        prefill_tps = float("nan")
        decode_tps = float("nan")
    else:
        ttft_v = ttft
        decode_wall = max(t_end - t_first, 1e-9)
        prefill_tps = (prompt_tokens / ttft_v) if ttft_v > 0 else float("nan")
        decode_tps = (n_gen / decode_wall) if n_gen > 0 else 0.0

    e2e = t_end - t_e2e0
    e2e_tps = (n_gen / e2e) if e2e > 0 and n_gen > 0 else float("nan")

    return GenerationMetrics(
        backend="mlxs",
        prompt_tokens=prompt_tokens,
        generated_tokens=n_gen,
        ttft_s=ttft_v,
        prefill_effective_tok_per_s=prefill_tps,
        decode_wall_s=decode_wall,
        decode_tok_per_s=decode_tps,
        end_to_end_wall_s=e2e,
        end_to_end_tok_per_s=e2e_tps,
        rss_bytes_after_generate=rss_bytes_self(),
        mlx_peak_memory_bytes=_mlx_peak_bytes(),
        error=None,
    )


def load_mlx_lm(
    model_path: Path,
    *,
    trust_remote_code: bool,
) -> LoadSession:
    from mlx_lm import load

    rss0 = rss_bytes_self()
    _reset_mlx_memory_stats()
    t0 = time.perf_counter()
    try:
        model, tokenizer = load(
            str(model_path),
            tokenizer_config={"trust_remote_code": trust_remote_code},
            lazy=False,
        )
    except Exception as exc:
        return LoadSession(
            backend="mlx_lm",
            model=None,
            tokenizer=None,
            load_wall_s=time.perf_counter() - t0,
            rss_bytes_before_load=rss0,
            rss_bytes_after_load=rss_bytes_self(),
            error=f"{type(exc).__name__}: {exc}",
        )
    return LoadSession(
        backend="mlx_lm",
        model=model,
        tokenizer=tokenizer,
        load_wall_s=time.perf_counter() - t0,
        rss_bytes_before_load=rss0,
        rss_bytes_after_load=rss_bytes_self(),
        error=None,
    )


def generate_mlx_lm_loaded(
    session: LoadSession,
    prompt_text: str,
    *,
    max_tokens: int,
    seed: int | None,
    prefill_step_size: int,
) -> GenerationMetrics:
    from mlx_lm.generate import stream_generate

    if session.error or session.model is None:
        return GenerationMetrics(
            backend="mlx_lm",
            prompt_tokens=0,
            generated_tokens=0,
            ttft_s=float("nan"),
            prefill_effective_tok_per_s=float("nan"),
            decode_wall_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_wall_s=float("nan"),
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=None,
            error=session.error or "no_model",
        )

    _reset_mlx_memory_stats()
    model = session.model
    tokenizer = session.tokenizer

    if seed is not None:
        mx.random.seed(seed)

    last = None
    t_e2e0 = time.perf_counter()
    try:
        # mlx-lm >=0.31: sampling kwargs are not forwarded to generate_step; greedy
        # uses the default argmax sampler inside generate_step.
        for resp in stream_generate(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
        ):
            last = resp
    except Exception as exc:
        return GenerationMetrics(
            backend="mlx_lm",
            prompt_tokens=0,
            generated_tokens=0,
            ttft_s=float("nan"),
            prefill_effective_tok_per_s=float("nan"),
            decode_wall_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_wall_s=time.perf_counter() - t_e2e0,
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=_mlx_peak_bytes(),
            error=f"{type(exc).__name__}: {exc}",
        )

    t_end = time.perf_counter()
    if last is None:
        return GenerationMetrics(
            backend="mlx_lm",
            prompt_tokens=0,
            generated_tokens=0,
            ttft_s=float("nan"),
            prefill_effective_tok_per_s=float("nan"),
            decode_wall_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_wall_s=t_end - t_e2e0,
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=_mlx_peak_bytes(),
            error="no_tokens",
        )

    pt = int(last.prompt_tokens)
    gt = int(last.generation_tokens)
    prompt_tps_lm = float(last.prompt_tps) if last.prompt_tps else float("nan")
    if prompt_tps_lm > 0 and not math.isnan(prompt_tps_lm):
        ttft_v = pt / prompt_tps_lm
    else:
        ttft_v = float("nan")
    decode_tps = float(last.generation_tps) if last.generation_tps else float("nan")
    decode_wall = (gt / decode_tps) if decode_tps > 0 else float("nan")
    e2e = t_end - t_e2e0
    e2e_tps = (gt / e2e) if e2e > 0 and gt > 0 else float("nan")

    return GenerationMetrics(
        backend="mlx_lm",
        prompt_tokens=pt,
        generated_tokens=gt,
        ttft_s=ttft_v,
        prefill_effective_tok_per_s=prompt_tps_lm,
        decode_wall_s=decode_wall,
        decode_tok_per_s=decode_tps,
        end_to_end_wall_s=e2e,
        end_to_end_tok_per_s=e2e_tps,
        rss_bytes_after_generate=rss_bytes_self(),
        mlx_peak_memory_bytes=_mlx_peak_bytes(),
        error=None,
    )


def close_session(session: LoadSession | None) -> None:
    if session is None or session.model is None:
        return
    _unload_best_effort(session.model, session.tokenizer)
