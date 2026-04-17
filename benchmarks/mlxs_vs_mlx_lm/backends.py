"""Backend load + explicit-boundary Class A generation helpers.

The canonical benchmark path is a pure Layer 1 benchmark:
MLXs uses the ``runtime_core`` subtree, while the mlx-lm baseline uses a
benchmark-local helper that mirrors the comparable upstream ``generate_step``
behavior without ``stream_generate`` detokenization and response shaping.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.mlxs_vs_mlx_lm.memory import rss_bytes_self

CANONICAL_BENCHMARK_CLASS = "Class A — Minimal fast-path decode"
CANONICAL_CLEAR_CACHE_INTERVAL = 256
CANONICAL_STREAM_POLICY = "dedicated_generation_stream"
CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD = 512
MLX_LM_UPSTREAM_ANCHOR = "mlx-lm main / release v0.31.2 (2026-04-07)"


def _require_mx() -> Any:
    import mlx.core as mx

    return mx


def _decode_tokens_measured(generated_tokens: int) -> int:
    return max(0, generated_tokens - 1)


def _tokens_per_second(tokens: int, wall_s: float) -> float:
    if tokens <= 0 or wall_s <= 0:
        return 0.0
    return tokens / wall_s


def runtime_profile(
    backend_variant: str,
    *,
    clear_cache_interval: int,
) -> dict[str, Any]:
    if backend_variant == "mlx_lm":
        return {
            "benchmark_class": CANONICAL_BENCHMARK_CLASS,
            "runtime_path_label": "benchmark-local helper mirroring mlx_lm.generate.generate_step",
            "runtime_root": "mlx_lm.generate.generate_step",
            "stream_handling": "Dedicated generation stream from mlx-lm upstream path.",
            "mx_async_eval": "Used during decode for current and next token buffers.",
            "mx_eval": "Explicit mx.eval(y) for the first generated token.",
            "item_extraction": "Yield path uses y.item() per generated token.",
            "clear_cache_decode_rule": (
                f"Loop index % {clear_cache_interval} == 0 after each yielded token "
                "(includes the first yielded token)."
            ),
            "upstream_anchor": MLX_LM_UPSTREAM_ANCHOR,
            "compile_policy": "Upstream eager comparable path.",
        }

    compile_policy = (
        "MLXs secondary compiled decode-forward variant."
        if backend_variant == "mlxs_compiled"
        else "MLXs eager canonical primary variant."
    )
    return {
        "benchmark_class": CANONICAL_BENCHMARK_CLASS,
        "runtime_path_label": (
            "benchmark-local helper on the canonical Layer 1 subtree with "
            "short-prompt prepared-step overlap"
        ),
        "runtime_root": (
            "mlxs.runtime_core.decode_step plus "
            "prepare_decode_step/schedule_next_decode_step/materialize_prepared_step "
            f"for prompts <= {CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD}"
        ),
        "layer1_subtree": [
            "CoreState.create/adopt",
            "run_prefill",
            "decode_step baseline for long prompts",
            "prepare_decode_step + mx.async_eval for short prompts",
            "schedule_next_decode_step + greedy_select for short prompts",
            "materialize_prepared_step + item() for short prompts",
            "termination.finish_for",
            "single-token forward lookahead step",
        ],
        "stream_handling": "Dedicated benchmark-owned generation stream via CoreExecutionPolicy.",
        "mx_async_eval": (
            "Used only on the short-prompt prepared-step branch "
            f"(prompt_tokens <= {CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD})."
        ),
        "mx_eval": (
            "Short prompts retain mx.eval(token) on the first generated token; "
            "long prompts use the baseline per-step mx.eval(token); "
            "cache state eval during chunked prefill."
        ),
        "item_extraction": "int(token.item()) per generated token.",
        "clear_cache_decode_rule": (
            f"generation_tokens % {clear_cache_interval} == 0 inside Layer 1 decode progression."
        ),
        "upstream_anchor": None,
        "compile_policy": compile_policy,
    }


@dataclass(slots=True)
class GenerationMetrics:
    """One canonical greedy generation after the model is already resident."""

    backend: str
    variant: str
    prompt_tokens: int
    generated_tokens: int
    decode_tokens_measured: int
    prefill_wall_s: float
    ttft_s: float
    decode_wall_s: float
    end_to_end_wall_s: float
    prefill_tok_per_s: float
    decode_tok_per_s: float
    end_to_end_tok_per_s: float
    rss_bytes_after_generate: int
    mlx_peak_memory_bytes: int | None
    error: str | None = None


@dataclass(slots=True)
class LoadSession:
    """Model resident in memory; reused for warmup + timed trials."""

    backend: str
    model: Any
    tokenizer: Any | None
    runtime_stream: Any | None
    load_wall_s: float
    rss_bytes_before_load: int
    rss_bytes_after_load: int
    error: str | None = None


def _reset_mlx_memory_stats() -> None:
    with contextlib.suppress(Exception):
        _require_mx().reset_peak_memory()


def _mlx_peak_bytes() -> int | None:
    try:
        return int(_require_mx().get_peak_memory())
    except Exception:
        return None


def _unload_best_effort(model: Any, tokenizer: Any | None) -> None:
    del model
    del tokenizer
    gc.collect()
    with contextlib.suppress(Exception):
        _require_mx().clear_cache()
    _reset_mlx_memory_stats()


def load_mlxs(
    model_path: Path,
    *,
    trust_remote_code: bool,
) -> LoadSession:
    del trust_remote_code
    mx = _require_mx()
    from mlxs._types import ModelMode
    from mlxs.load.loader import load_model

    rss0 = rss_bytes_self()
    _reset_mlx_memory_stats()
    t0 = time.perf_counter()
    try:
        model = load_model(
            model_path,
            lazy=False,
            model_mode=ModelMode.AUTO,
        )
        runtime_stream = mx.new_stream(mx.default_device())
    except Exception as exc:
        return LoadSession(
            backend="mlxs",
            model=None,
            tokenizer=None,
            runtime_stream=None,
            load_wall_s=time.perf_counter() - t0,
            rss_bytes_before_load=rss0,
            rss_bytes_after_load=rss_bytes_self(),
            error=f"{type(exc).__name__}: {exc}",
        )
    return LoadSession(
        backend="mlxs",
        model=model,
        tokenizer=None,
        runtime_stream=runtime_stream,
        load_wall_s=time.perf_counter() - t0,
        rss_bytes_before_load=rss0,
        rss_bytes_after_load=rss_bytes_self(),
        error=None,
    )


def _run_mlxs_class_a_trial(
    session: LoadSession,
    prompt_token_ids: Sequence[int],
    *,
    max_tokens: int,
    seed: int,
    prefill_step_size: int,
    clear_cache_interval: int,
    compile_decode: bool,
) -> GenerationMetrics:
    mx = _require_mx()
    from mlxs.runtime_core import CoreExecutionPolicy, CoreState, CoreTerminationPolicy
    from mlxs.runtime_core.decode import (
        decode_step,
        materialize_prepared_step,
        prepare_decode_step,
        schedule_next_decode_step,
    )
    from mlxs.runtime_core.prefill import run_prefill

    variant = "mlxs_compiled" if compile_decode else "mlxs_eager"
    if session.error or session.model is None:
        return GenerationMetrics(
            backend="mlxs",
            variant=variant,
            prompt_tokens=0,
            generated_tokens=0,
            decode_tokens_measured=0,
            prefill_wall_s=float("nan"),
            ttft_s=float("nan"),
            decode_wall_s=float("nan"),
            end_to_end_wall_s=float("nan"),
            prefill_tok_per_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=None,
            error=session.error or "no_model",
        )

    _reset_mlx_memory_stats()
    model = session.model
    mx.random.seed(seed)
    prompt_tokens = tuple(int(token_id) for token_id in prompt_token_ids)
    prompt_array = mx.array(prompt_tokens)
    execution = CoreExecutionPolicy(
        clear_cache_interval=clear_cache_interval,
        stream=session.runtime_stream,
    )
    termination = CoreTerminationPolicy(max_tokens=max_tokens, eos_token_ids=())

    if compile_decode:
        cache = model.make_cache()
        state = CoreState.adopt(cache)

        @mx.compile
        def step_fn(input_ids: Any) -> Any:
            return model(input_ids, cache=cache)

    else:
        state = CoreState.create(model)

        def step_fn(input_ids: Any) -> Any:
            return model(input_ids, cache=state.cache)

    t_start = time.perf_counter()
    generated_tokens = 0
    try:
        logits = run_prefill(
            model,
            prompt_array,
            state,
            execution=execution,
            prefill_step_size=prefill_step_size,
        )
        t_prefill_end = time.perf_counter()
        t_first: float | None = None
        t_end: float | None = None
        use_short_prompt_lookahead = (
            len(prompt_tokens) <= CANONICAL_SHORT_PROMPT_LOOKAHEAD_THRESHOLD
        )

        if use_short_prompt_lookahead:
            prepared = prepare_decode_step(logits, execution=execution, prime_token=True)

            while True:
                next_prepared = None
                # Class A uses a fixed length budget with no EOS tokens, so a
                # single-token lookahead stays within the canonical contract.
                if generated_tokens + 1 < max_tokens:
                    next_prepared = schedule_next_decode_step(
                        model,
                        state,
                        prepared,
                        execution=execution,
                        step_fn=step_fn,
                        prime_token=True,
                    )

                result = materialize_prepared_step(
                    state,
                    prepared,
                    termination=termination,
                    execution=execution,
                    force_eval=generated_tokens == 0,
                )
                generated_tokens = result.generation_tokens
                now = time.perf_counter()
                if t_first is None:
                    t_first = now

                if result.finish is not None:
                    t_end = now
                    break

                assert next_prepared is not None
                prepared = next_prepared
        else:
            while True:
                result, next_logits = decode_step(
                    model,
                    state,
                    logits,
                    termination=termination,
                    execution=execution,
                    step_fn=step_fn,
                )
                generated_tokens = result.generation_tokens
                now = time.perf_counter()
                if t_first is None:
                    t_first = now

                if result.finish is not None:
                    t_end = now
                    break

                assert next_logits is not None
                logits = next_logits

        assert t_first is not None
        assert t_end is not None
    except Exception as exc:
        return GenerationMetrics(
            backend="mlxs",
            variant=variant,
            prompt_tokens=len(prompt_tokens),
            generated_tokens=generated_tokens,
            decode_tokens_measured=_decode_tokens_measured(generated_tokens),
            prefill_wall_s=float("nan"),
            ttft_s=float("nan"),
            decode_wall_s=float("nan"),
            end_to_end_wall_s=time.perf_counter() - t_start,
            prefill_tok_per_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=_mlx_peak_bytes(),
            error=f"{type(exc).__name__}: {exc}",
        )

    prefill_wall_s = t_prefill_end - t_start
    ttft_s = t_first - t_start
    decode_wall_s = max(0.0, t_end - t_first)
    end_to_end_wall_s = t_end - t_start
    decode_tokens_measured = _decode_tokens_measured(generated_tokens)

    return GenerationMetrics(
        backend="mlxs",
        variant=variant,
        prompt_tokens=len(prompt_tokens),
        generated_tokens=generated_tokens,
        decode_tokens_measured=decode_tokens_measured,
        prefill_wall_s=prefill_wall_s,
        ttft_s=ttft_s,
        decode_wall_s=decode_wall_s,
        end_to_end_wall_s=end_to_end_wall_s,
        prefill_tok_per_s=_tokens_per_second(len(prompt_tokens), prefill_wall_s),
        decode_tok_per_s=_tokens_per_second(decode_tokens_measured, decode_wall_s),
        end_to_end_tok_per_s=_tokens_per_second(generated_tokens, end_to_end_wall_s),
        rss_bytes_after_generate=rss_bytes_self(),
        mlx_peak_memory_bytes=_mlx_peak_bytes(),
        error=None,
    )


def generate_mlxs_loaded(
    session: LoadSession,
    prompt_token_ids: Sequence[int],
    *,
    max_tokens: int,
    seed: int,
    prefill_step_size: int,
    clear_cache_interval: int,
    compile_decode: bool,
) -> GenerationMetrics:
    return _run_mlxs_class_a_trial(
        session,
        prompt_token_ids,
        max_tokens=max_tokens,
        seed=seed,
        prefill_step_size=prefill_step_size,
        clear_cache_interval=clear_cache_interval,
        compile_decode=compile_decode,
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
            runtime_stream=None,
            load_wall_s=time.perf_counter() - t0,
            rss_bytes_before_load=rss0,
            rss_bytes_after_load=rss_bytes_self(),
            error=f"{type(exc).__name__}: {exc}",
        )
    return LoadSession(
        backend="mlx_lm",
        model=model,
        tokenizer=tokenizer,
        runtime_stream=None,
        load_wall_s=time.perf_counter() - t0,
        rss_bytes_before_load=rss0,
        rss_bytes_after_load=rss_bytes_self(),
        error=None,
    )


def _run_mlx_lm_class_a_trial(
    session: LoadSession,
    prompt_token_ids: Sequence[int],
    *,
    max_tokens: int,
    seed: int | None,
    prefill_step_size: int,
    clear_cache_interval: int,
) -> GenerationMetrics:
    mx = _require_mx()
    import importlib

    mlx_lm_generate = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models import cache as mlx_lm_cache

    if session.error or session.model is None:
        return GenerationMetrics(
            backend="mlx_lm",
            variant="mlx_lm",
            prompt_tokens=0,
            generated_tokens=0,
            decode_tokens_measured=0,
            prefill_wall_s=float("nan"),
            ttft_s=float("nan"),
            decode_wall_s=float("nan"),
            end_to_end_wall_s=float("nan"),
            prefill_tok_per_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=None,
            error=session.error or "no_model",
        )

    _reset_mlx_memory_stats()
    model = session.model
    if seed is not None:
        mx.random.seed(seed)

    prompt_tokens = tuple(int(token_id) for token_id in prompt_token_ids)
    prompt = mx.array(prompt_tokens)
    prompt_cache = mlx_lm_cache.make_prompt_cache(model, max_kv_size=None)
    generation_stream = mlx_lm_generate.generation_stream

    def _model_call(input_tokens: Any) -> Any:
        return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: Any) -> tuple[Any, Any]:
        with mx.stream(generation_stream):
            logits = _model_call(input_tokens[None])[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            sampled = mx.argmax(logprobs, axis=-1)
        return sampled, logprobs.squeeze(0)

    t_start = time.perf_counter()
    generated_tokens = 0
    try:
        working_prompt = prompt
        with mx.stream(generation_stream):
            while working_prompt.size > 1:
                n_to_process = min(prefill_step_size, int(working_prompt.size) - 1)
                _model_call(working_prompt[:n_to_process][None])
                cache_states = [
                    cache_state
                    for cache_state in (getattr(cache, "state", None) for cache in prompt_cache)
                    if cache_state is not None
                ]
                if cache_states:
                    mx.eval(cache_states)
                working_prompt = working_prompt[n_to_process:]
                mx.clear_cache()

            y, logprobs = _step(working_prompt)
            mx.async_eval(y, logprobs)
        t_prefill_end = time.perf_counter()

        t_first: float | None = None
        t_end: float | None = None
        while generated_tokens < max_tokens:
            next_y: Any | None = None
            next_logprobs: Any | None = None
            if generated_tokens != max_tokens:
                next_y, next_logprobs = _step(y)
                mx.async_eval(next_y, next_logprobs)
            if generated_tokens == 0:
                mx.eval(y)

            token_id = int(y.item())
            del token_id
            now = time.perf_counter()
            generated_tokens += 1
            if t_first is None:
                t_first = now

            if clear_cache_interval > 0 and (generated_tokens - 1) % clear_cache_interval == 0:
                mx.clear_cache()

            if generated_tokens >= max_tokens:
                t_end = now
                break

            assert next_y is not None
            assert next_logprobs is not None
            y, logprobs = next_y, next_logprobs

        assert t_first is not None
        assert t_end is not None
    except Exception as exc:
        return GenerationMetrics(
            backend="mlx_lm",
            variant="mlx_lm",
            prompt_tokens=len(prompt_tokens),
            generated_tokens=generated_tokens,
            decode_tokens_measured=_decode_tokens_measured(generated_tokens),
            prefill_wall_s=float("nan"),
            ttft_s=float("nan"),
            decode_wall_s=float("nan"),
            end_to_end_wall_s=time.perf_counter() - t_start,
            prefill_tok_per_s=float("nan"),
            decode_tok_per_s=float("nan"),
            end_to_end_tok_per_s=float("nan"),
            rss_bytes_after_generate=rss_bytes_self(),
            mlx_peak_memory_bytes=_mlx_peak_bytes(),
            error=f"{type(exc).__name__}: {exc}",
        )

    prefill_wall_s = t_prefill_end - t_start
    ttft_s = t_first - t_start
    decode_wall_s = max(0.0, t_end - t_first)
    end_to_end_wall_s = t_end - t_start
    decode_tokens_measured = _decode_tokens_measured(generated_tokens)

    return GenerationMetrics(
        backend="mlx_lm",
        variant="mlx_lm",
        prompt_tokens=len(prompt_tokens),
        generated_tokens=generated_tokens,
        decode_tokens_measured=decode_tokens_measured,
        prefill_wall_s=prefill_wall_s,
        ttft_s=ttft_s,
        decode_wall_s=decode_wall_s,
        end_to_end_wall_s=end_to_end_wall_s,
        prefill_tok_per_s=_tokens_per_second(len(prompt_tokens), prefill_wall_s),
        decode_tok_per_s=_tokens_per_second(decode_tokens_measured, decode_wall_s),
        end_to_end_tok_per_s=_tokens_per_second(generated_tokens, end_to_end_wall_s),
        rss_bytes_after_generate=rss_bytes_self(),
        mlx_peak_memory_bytes=_mlx_peak_bytes(),
        error=None,
    )


def generate_mlx_lm_loaded(
    session: LoadSession,
    prompt_token_ids: Sequence[int],
    *,
    max_tokens: int,
    seed: int | None,
    prefill_step_size: int,
    clear_cache_interval: int,
) -> GenerationMetrics:
    return _run_mlx_lm_class_a_trial(
        session,
        prompt_token_ids,
        max_tokens=max_tokens,
        seed=seed,
        prefill_step_size=prefill_step_size,
        clear_cache_interval=clear_cache_interval,
    )


def close_session(session: LoadSession | None) -> None:
    if session is None or session.model is None:
        return
    _unload_best_effort(session.model, session.tokenizer)
