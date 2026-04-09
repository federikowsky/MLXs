"""Decode Engine V4 flat core generator."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache import convert_to_quantized
from mlxs.cache.kv import KVCache
from mlxs.generate.prefill import chunked_prefill
from mlxs.generate.recipe import (
    CompileMode,
    ExplicitOffsets,
    ExplicitState,
    LogitsStepFn,
    Recipe,
    build_explicit_state_step_fn,
    build_logits_step_fn,
    build_resident_greedy_session,
    build_step_fn,
    flatten_explicit_state,
    unflatten_explicit_state,
)

if TYPE_CHECKING:
    from mlxs.generate.profile import DecodeProfiler

_generation_stream = mx.new_stream(mx.default_device())


def require_async_eval() -> Callable[..., object]:
    """Return ``mx.async_eval`` or raise if it is unavailable."""
    async_eval = getattr(mx, "async_eval", None)
    if async_eval is None:
        raise RuntimeError("mx.async_eval is required by Decode Engine V4")
    return cast(Callable[..., object], async_eval)


def _decode_steps(
    model: nn.Module,
    cache: list[KVCache],
    recipe: Recipe,
    stream: mx.Stream,
    first_logits: mx.array,
    max_tokens: int,
    clear_cache_interval: int,
    *,
    quantized_kv_start: int,
    kv_bits: int | None,
    kv_group_size: int,
) -> Generator[tuple[int, mx.array], None, None]:
    """Yield raw token ids and lazy logprobs for the supported M1 recipe."""
    if recipe.has_processors:
        yield from _decode_steps_with_processors(
            model,
            cache,
            recipe,
            stream,
            first_logits,
            max_tokens,
            clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )
        return

    async_eval = require_async_eval()
    step_fn = build_step_fn(model, cache, recipe, stream)

    with mx.stream(stream):
        seed_logprobs_2d = first_logits - mx.logsumexp(first_logits, keepdims=True)
        seed_token = recipe.sampler(seed_logprobs_2d)
        seed_logprobs = seed_logprobs_2d.squeeze(0)

    async_eval(seed_token)
    mx.eval(seed_token)
    yield int(seed_token.item()), seed_logprobs
    kv_already_quantized = _apply_eager_mutation_boundary(
        cache,
        yielded_count=1,
        clear_cache_interval=clear_cache_interval,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_already_quantized=False,
    )

    if max_tokens <= 1:
        return

    current_token, current_logprobs = step_fn(seed_token)
    async_eval(current_token)
    yielded_count = 1

    while True:
        if yielded_count + 1 == max_tokens:
            yield int(current_token.item()), current_logprobs
            return

        next_token, next_logprobs = step_fn(current_token)
        async_eval(next_token)
        yield int(current_token.item()), current_logprobs

        yielded_count += 1
        kv_already_quantized = _apply_eager_mutation_boundary(
            cache,
            yielded_count=yielded_count,
            clear_cache_interval=clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            kv_already_quantized=kv_already_quantized,
        )
        current_token, current_logprobs = next_token, next_logprobs


def _decode_steps_detail(
    model: nn.Module,
    cache: list[KVCache],
    recipe: Recipe,
    stream: mx.Stream,
    first_logits: mx.array,
    max_tokens: int,
    clear_cache_interval: int,
    profiler: DecodeProfiler,
    *,
    quantized_kv_start: int,
    kv_bits: int | None,
    kv_group_size: int,
) -> Generator[tuple[int, mx.array], None, None]:
    """Perturbative detail-mode variant of the V4 core generator."""
    if recipe.has_processors:
        yield from _decode_steps_with_processors_detail(
            model,
            cache,
            recipe,
            stream,
            first_logits,
            max_tokens,
            clear_cache_interval,
            profiler,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )
        return

    async_eval = require_async_eval()
    step_fn = build_step_fn(model, cache, recipe, stream)

    with mx.stream(stream):
        seed_logprobs_2d = first_logits - mx.logsumexp(first_logits, keepdims=True)
        seed_token = recipe.sampler(seed_logprobs_2d)
        seed_logprobs = seed_logprobs_2d.squeeze(0)

    t0 = time.perf_counter()
    async_eval(seed_token)
    profiler.async_eval_wall_s.append(time.perf_counter() - t0)
    mx.eval(seed_token)

    t0 = time.perf_counter()
    seed_token_id = int(seed_token.item())
    profiler.item_wait_wall_s.append(time.perf_counter() - t0)
    yield seed_token_id, seed_logprobs
    kv_already_quantized = _apply_eager_mutation_boundary(
        cache,
        yielded_count=1,
        clear_cache_interval=clear_cache_interval,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_already_quantized=False,
    )

    if max_tokens <= 1:
        return

    t0 = time.perf_counter()
    current_token, current_logprobs = step_fn(seed_token)
    profiler.step_fn_wall_s.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    async_eval(current_token)
    profiler.async_eval_wall_s.append(time.perf_counter() - t0)
    yielded_count = 1

    while True:
        if yielded_count + 1 == max_tokens:
            t0 = time.perf_counter()
            current_token_id = int(current_token.item())
            profiler.item_wait_wall_s.append(time.perf_counter() - t0)
            yield current_token_id, current_logprobs
            return

        t0 = time.perf_counter()
        next_token, next_logprobs = step_fn(current_token)
        profiler.step_fn_wall_s.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        async_eval(next_token)
        profiler.async_eval_wall_s.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        current_token_id = int(current_token.item())
        profiler.item_wait_wall_s.append(time.perf_counter() - t0)
        yield current_token_id, current_logprobs

        yielded_count += 1
        kv_already_quantized = _apply_eager_mutation_boundary(
            cache,
            yielded_count=yielded_count,
            clear_cache_interval=clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            kv_already_quantized=kv_already_quantized,
        )
        current_token, current_logprobs = next_token, next_logprobs


def _decode_steps_explicit_state(
    model: nn.Module,
    cache: list[KVCache],
    recipe: Recipe,
    stream: mx.Stream,
    first_logits: mx.array,
    max_tokens: int,
    clear_cache_interval: int,
    *,
    final_cache_holder: list[list[KVCache] | None] | None = None,
) -> Generator[tuple[int, mx.array], None, None]:
    """Experimental P2 public-path core for compile-on with explicit state."""
    if recipe.compile_mode is not CompileMode.ON:
        raise NotImplementedError("Explicit-state decode requires compile mode on")

    async_eval = require_async_eval()

    history_tokens: mx.array | None = None
    seed_token, seed_logprobs, history_tokens = _sample_from_logits(
        first_logits,
        history_tokens,
        recipe,
        stream,
    )

    state_arrays: ExplicitState | None = None
    offsets: ExplicitOffsets | None = None
    snapshot_state = None
    try:
        async_eval(seed_token)
        mx.eval(seed_token)
        yield int(seed_token.item()), seed_logprobs

        if max_tokens <= 1:
            return

        prompt_slots = max(layer_cache.offset for layer_cache in cache)
        total_slots = prompt_slots + max_tokens
        state_arrays = export_kvcache_to_explicit_state(cache, total_slots=total_slots)
        _, _, offsets = unflatten_explicit_state(state_arrays, num_layers=len(cache))
        step_fn, snapshot_state = build_explicit_state_step_fn(
            model,
            recipe,
            stream,
            num_layers=len(cache),
            initial_state=state_arrays,
        )

        if clear_cache_interval > 0 and 0 % clear_cache_interval == 0:
            mx.clear_cache()

        current_logits_or_logprobs, *new_offsets = step_fn(seed_token, *offsets)
        offsets = cast(ExplicitOffsets, tuple(new_offsets))
        if recipe.has_processors:
            current_token, current_logprobs, history_tokens = _sample_from_logits(
                current_logits_or_logprobs,
                history_tokens,
                recipe,
                stream,
            )
        else:
            current_logprobs = current_logits_or_logprobs
            with mx.stream(stream):
                current_token = recipe.sampler(current_logprobs[None, :])
            history_tokens = _append_processor_history(
                history_tokens,
                current_token,
                recipe.processor_context_size,
            )
        async_eval(current_token)
        yielded_count = 1

        while True:
            if yielded_count + 1 == max_tokens:
                yield int(current_token.item()), current_logprobs
                return

            next_logits_or_logprobs, *new_offsets = step_fn(current_token, *offsets)
            offsets = cast(ExplicitOffsets, tuple(new_offsets))
            if recipe.has_processors:
                next_token, next_logprobs, history_tokens = _sample_from_logits(
                    next_logits_or_logprobs,
                    history_tokens,
                    recipe,
                    stream,
                )
            else:
                next_logprobs = next_logits_or_logprobs
                with mx.stream(stream):
                    next_token = recipe.sampler(next_logprobs[None, :])
                history_tokens = _append_processor_history(
                    history_tokens,
                    next_token,
                    recipe.processor_context_size,
                )
            async_eval(next_token)
            yield int(current_token.item()), current_logprobs

            yielded_count += 1
            if clear_cache_interval > 0 and (yielded_count - 1) % clear_cache_interval == 0:
                mx.clear_cache()
            current_token, current_logprobs = next_token, next_logprobs
    finally:
        if final_cache_holder is not None:
            final_cache_holder[:] = [
                cache
                if state_arrays is None or offsets is None or snapshot_state is None
                else materialize_explicit_state_to_kvcache(
                    snapshot_state(offsets),
                    num_layers=len(cache),
                )
            ]


def _append_processor_history(
    history_tokens: mx.array | None,
    token: mx.array,
    context_size: int,
) -> mx.array:
    token_1d = token.astype(mx.int32).reshape(-1)
    if history_tokens is None:
        updated = token_1d
    else:
        updated = mx.concatenate([history_tokens, token_1d], axis=0)
    if context_size > 0 and updated.shape[0] > context_size:
        return updated[-context_size:]
    return updated


def _apply_logits_processors(
    logits: mx.array,
    history_tokens: mx.array | None,
    recipe: Recipe,
) -> mx.array:
    logits_2d = logits if logits.ndim == 2 else logits[None, :]
    if not recipe.has_processors:
        return logits_2d

    tokens = history_tokens
    if tokens is None:
        tokens = mx.array([], dtype=mx.int32)

    processed = logits_2d
    for processor in recipe.logits_processors:
        processor_fn = cast(Callable[[mx.array, mx.array], mx.array], processor)
        processed = processor_fn(tokens, processed)
    return processed


def _sample_from_logits(
    logits: mx.array,
    history_tokens: mx.array | None,
    recipe: Recipe,
    stream: mx.Stream,
) -> tuple[mx.array, mx.array, mx.array]:
    with mx.stream(stream):
        processed_logits = _apply_logits_processors(logits, history_tokens, recipe)
        logprobs_2d = processed_logits - mx.logsumexp(processed_logits, axis=-1, keepdims=True)
        next_token = recipe.sampler(logprobs_2d)
    next_history = _append_processor_history(
        history_tokens,
        next_token,
        recipe.processor_context_size,
    )
    return next_token, logprobs_2d.squeeze(0), next_history


def _decode_steps_with_processors(
    model: nn.Module,
    cache: list[KVCache],
    recipe: Recipe,
    stream: mx.Stream,
    first_logits: mx.array,
    max_tokens: int,
    clear_cache_interval: int,
    *,
    quantized_kv_start: int,
    kv_bits: int | None,
    kv_group_size: int,
) -> Generator[tuple[int, mx.array], None, None]:
    async_eval = require_async_eval()
    step_fn: LogitsStepFn = build_logits_step_fn(model, cache, recipe, stream)
    history_tokens: mx.array | None = None

    seed_token, seed_logprobs, history_tokens = _sample_from_logits(
        first_logits,
        history_tokens,
        recipe,
        stream,
    )
    async_eval(seed_token)
    mx.eval(seed_token)
    yield int(seed_token.item()), seed_logprobs
    kv_already_quantized = _apply_eager_mutation_boundary(
        cache,
        yielded_count=1,
        clear_cache_interval=clear_cache_interval,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_already_quantized=False,
    )

    if max_tokens <= 1:
        return

    current_logits = step_fn(seed_token)
    current_token, current_logprobs, history_tokens = _sample_from_logits(
        current_logits,
        history_tokens,
        recipe,
        stream,
    )
    async_eval(current_token)
    yielded_count = 1

    while True:
        if yielded_count + 1 == max_tokens:
            yield int(current_token.item()), current_logprobs
            return

        next_logits = step_fn(current_token)
        next_token, next_logprobs, history_tokens = _sample_from_logits(
            next_logits,
            history_tokens,
            recipe,
            stream,
        )
        async_eval(next_token)
        yield int(current_token.item()), current_logprobs

        yielded_count += 1
        kv_already_quantized = _apply_eager_mutation_boundary(
            cache,
            yielded_count=yielded_count,
            clear_cache_interval=clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            kv_already_quantized=kv_already_quantized,
        )
        current_token, current_logprobs = next_token, next_logprobs


def _decode_steps_with_processors_detail(
    model: nn.Module,
    cache: list[KVCache],
    recipe: Recipe,
    stream: mx.Stream,
    first_logits: mx.array,
    max_tokens: int,
    clear_cache_interval: int,
    profiler: DecodeProfiler,
    *,
    quantized_kv_start: int,
    kv_bits: int | None,
    kv_group_size: int,
) -> Generator[tuple[int, mx.array], None, None]:
    async_eval = require_async_eval()
    step_fn: LogitsStepFn = build_logits_step_fn(model, cache, recipe, stream)
    history_tokens: mx.array | None = None

    seed_token, seed_logprobs, history_tokens = _sample_from_logits(
        first_logits,
        history_tokens,
        recipe,
        stream,
    )
    t0 = time.perf_counter()
    async_eval(seed_token)
    profiler.async_eval_wall_s.append(time.perf_counter() - t0)
    mx.eval(seed_token)

    t0 = time.perf_counter()
    seed_token_id = int(seed_token.item())
    profiler.item_wait_wall_s.append(time.perf_counter() - t0)
    yield seed_token_id, seed_logprobs
    kv_already_quantized = _apply_eager_mutation_boundary(
        cache,
        yielded_count=1,
        clear_cache_interval=clear_cache_interval,
        quantized_kv_start=quantized_kv_start,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_already_quantized=False,
    )

    if max_tokens <= 1:
        return

    t0 = time.perf_counter()
    current_logits = step_fn(seed_token)
    profiler.step_fn_wall_s.append(time.perf_counter() - t0)
    current_token, current_logprobs, history_tokens = _sample_from_logits(
        current_logits,
        history_tokens,
        recipe,
        stream,
    )
    t0 = time.perf_counter()
    async_eval(current_token)
    profiler.async_eval_wall_s.append(time.perf_counter() - t0)
    yielded_count = 1

    while True:
        if yielded_count + 1 == max_tokens:
            t0 = time.perf_counter()
            current_token_id = int(current_token.item())
            profiler.item_wait_wall_s.append(time.perf_counter() - t0)
            yield current_token_id, current_logprobs
            return

        t0 = time.perf_counter()
        next_logits = step_fn(current_token)
        profiler.step_fn_wall_s.append(time.perf_counter() - t0)
        next_token, next_logprobs, history_tokens = _sample_from_logits(
            next_logits,
            history_tokens,
            recipe,
            stream,
        )

        t0 = time.perf_counter()
        async_eval(next_token)
        profiler.async_eval_wall_s.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        current_token_id = int(current_token.item())
        profiler.item_wait_wall_s.append(time.perf_counter() - t0)
        yield current_token_id, current_logprobs

        yielded_count += 1
        kv_already_quantized = _apply_eager_mutation_boundary(
            cache,
            yielded_count=yielded_count,
            clear_cache_interval=clear_cache_interval,
            quantized_kv_start=quantized_kv_start,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            kv_already_quantized=kv_already_quantized,
        )
        current_token, current_logprobs = next_token, next_logprobs


def _apply_eager_mutation_boundary(
    cache: list[KVCache],
    *,
    yielded_count: int,
    clear_cache_interval: int,
    quantized_kv_start: int,
    kv_bits: int | None,
    kv_group_size: int,
    kv_already_quantized: bool,
) -> bool:
    if clear_cache_interval > 0 and (yielded_count - 1) % clear_cache_interval == 0:
        mx.clear_cache()

    if (
        not kv_already_quantized
        and quantized_kv_start > 0
        and kv_bits is not None
        and yielded_count == quantized_kv_start
    ):
        cache[:] = cast(
            list[KVCache],
            convert_to_quantized(
                cache,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
            ),
        )
        return True

    return kv_already_quantized

def export_kvcache_to_explicit_state(
    cache: list[KVCache],
    *,
    total_slots: int,
) -> ExplicitState:
    """Export filled ``KVCache`` state into fixed-capacity explicit arrays."""
    k_arrays: list[mx.array] = []
    v_arrays: list[mx.array] = []
    offsets: list[mx.array] = []

    for layer_cache in cache:
        state = layer_cache.state
        if state is None:
            raise ValueError("Experimental P2 explicit-state export requires a non-empty KV cache")
        keys, values = state
        used_slots = keys.shape[2]
        if used_slots > total_slots:
            raise ValueError(
                f"KV cache uses {used_slots} slots but explicit-state capacity is {total_slots}"
            )

        tail_slots = total_slots - used_slots
        if tail_slots > 0:
            k_tail = mx.zeros(
                (*keys.shape[:2], tail_slots, keys.shape[3]),
                dtype=keys.dtype,
            )
            v_tail = mx.zeros(
                (*values.shape[:2], tail_slots, values.shape[3]),
                dtype=values.dtype,
            )
            full_keys = mx.concatenate([keys, k_tail], axis=2)
            full_values = mx.concatenate([values, v_tail], axis=2)
        else:
            full_keys = keys
            full_values = values

        k_arrays.append(full_keys)
        v_arrays.append(full_values)
        offsets.append(mx.array([used_slots], dtype=mx.int32))

    return flatten_explicit_state(tuple(k_arrays), tuple(v_arrays), tuple(offsets))


def materialize_explicit_state_to_kvcache(
    state_arrays: ExplicitState,
    *,
    num_layers: int,
) -> list[KVCache]:
    """Materialize explicit state back into ``KVCache`` objects at the boundary."""
    k_arrays, v_arrays, offsets = unflatten_explicit_state(state_arrays, num_layers=num_layers)
    caches: list[KVCache] = []
    for k_array, v_array, offset in zip(k_arrays, v_arrays, offsets, strict=True):
        used_slots = int(offset.item())
        cache = KVCache()
        cache.state = (
            k_array[..., :used_slots, :],
            v_array[..., :used_slots, :],
        )
        caches.append(cache)
    return caches


def run_explicit_state_decode_experiment(
    model: nn.Module,
    prompt_tokens: list[int],
    recipe: Recipe,
    *,
    max_tokens: int,
    prefill_step_size: int = 2048,
    input_embeddings: mx.array | None = None,
    stream: mx.Stream = _generation_stream,
) -> tuple[list[int], list[KVCache]]:
    """Run the experimental P2 decode path outside the shipping generator."""
    if recipe.compile_mode is not CompileMode.ON:
        raise NotImplementedError("Experimental P2 decode requires compile mode on")
    if not prompt_tokens:
        raise ValueError("Experimental P2 decode requires a non-empty prompt")
    if max_tokens <= 0:
        return [], []

    prefill_cache = model.make_cache()
    prompt_array = mx.array(prompt_tokens)
    first_logits = chunked_prefill(
        model,
        prompt_array,
        prefill_cache,
        prefill_step_size=prefill_step_size,
        input_embeddings=input_embeddings,
        stream=stream,
    )

    history_tokens: mx.array | None = None
    seed_token, _, history_tokens = _sample_from_logits(
        first_logits,
        history_tokens,
        recipe,
        stream,
    )
    mx.eval(seed_token)

    token_ids = [int(seed_token.item())]
    if max_tokens == 1:
        return token_ids, prefill_cache

    total_slots = len(prompt_tokens) + max_tokens
    state_arrays = export_kvcache_to_explicit_state(prefill_cache, total_slots=total_slots)
    _, _, offsets = unflatten_explicit_state(state_arrays, num_layers=len(prefill_cache))
    step_fn, snapshot_state = build_explicit_state_step_fn(
        model,
        recipe,
        stream,
        num_layers=len(prefill_cache),
        initial_state=state_arrays,
    )

    prev_token = seed_token
    for _ in range(1, max_tokens):
        logits_or_logprobs, *new_offsets = step_fn(prev_token, *offsets)
        offsets = cast(ExplicitOffsets, tuple(new_offsets))
        if recipe.has_processors:
            next_token, _, history_tokens = _sample_from_logits(
                logits_or_logprobs,
                history_tokens,
                recipe,
                stream,
            )
        else:
            logprobs = logits_or_logprobs
            with mx.stream(stream):
                next_token = recipe.sampler(logprobs[None, :])
            history_tokens = _append_processor_history(
                history_tokens,
                next_token,
                recipe.processor_context_size,
            )
        mx.eval(next_token)
        token_ids.append(int(next_token.item()))
        prev_token = next_token

    return token_ids, materialize_explicit_state_to_kvcache(
        snapshot_state(offsets),
        num_layers=len(prefill_cache),
    )


def run_resident_state_decode_experiment(
    model: nn.Module,
    prompt_tokens: list[int],
    recipe: Recipe,
    *,
    max_tokens: int,
    eos_token_id: int | None = None,
    prefill_step_size: int = 2048,
    input_embeddings: mx.array | None = None,
    stream: mx.Stream = _generation_stream,
) -> tuple[list[int], list[KVCache]]:
    """Run the resident-state greedy slice outside the shipping generator."""
    if recipe.compile_mode is not CompileMode.ON:
        raise NotImplementedError("Resident-state decode requires compile mode on")
    if recipe.has_processors or recipe.logits_processors:
        raise NotImplementedError("Resident-state decode currently supports greedy only")
    if recipe.emit_logprobs or recipe.emit_top_logprobs:
        raise NotImplementedError("Resident-state decode currently does not emit logprobs")
    if not prompt_tokens:
        raise ValueError("Resident-state decode requires a non-empty prompt")
    if max_tokens <= 0:
        return [], []

    prefill_cache = model.make_cache()
    prompt_array = mx.array(prompt_tokens)
    first_logits = chunked_prefill(
        model,
        prompt_array,
        prefill_cache,
        prefill_step_size=prefill_step_size,
        input_embeddings=input_embeddings,
        stream=stream,
    )

    with mx.stream(stream):
        seed_logprobs = first_logits - mx.logsumexp(first_logits, axis=-1, keepdims=True)
        seed_token = recipe.sampler(seed_logprobs).astype(mx.int32).reshape(1)
    mx.eval(seed_token)

    token_ids = [int(seed_token.item())]
    if max_tokens == 1 or (eos_token_id is not None and token_ids[-1] == eos_token_id):
        return token_ids, prefill_cache

    total_slots = len(prompt_tokens) + max_tokens
    state_arrays = export_kvcache_to_explicit_state(prefill_cache, total_slots=total_slots)
    session = build_resident_greedy_session(
        model,
        recipe,
        stream,
        num_layers=len(prefill_cache),
        initial_state=state_arrays,
        seed_token=seed_token,
    )

    for _ in range(1, max_tokens):
        next_token = session.advance()
        mx.eval(next_token)
        token_id = int(next_token.item())
        token_ids.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break

    return token_ids, materialize_explicit_state_to_kvcache(
        session.snapshot_state(),
        num_layers=len(prefill_cache),
    )


__all__ = [
    "_decode_steps",
    "_decode_steps_detail",
    "_decode_steps_explicit_state",
    "_generation_stream",
    "export_kvcache_to_explicit_state",
    "materialize_explicit_state_to_kvcache",
    "require_async_eval",
    "run_explicit_state_decode_experiment",
    "run_resident_state_decode_experiment",
]
