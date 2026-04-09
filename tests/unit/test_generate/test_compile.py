"""Synthetic compile/cache protocol probes for V5 M3b P1."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import mlx.core as mx
import pytest

from mlxs.cache.kv import KVCache
from mlxs.generate.core import (
    _generation_stream,
    export_kvcache_to_explicit_state,
    materialize_explicit_state_to_kvcache,
    run_explicit_state_decode_experiment,
    run_resident_state_decode_experiment,
)
from mlxs.generate.recipe import (
    CompileMode,
    Recipe,
    build_explicit_state_step_fn,
    build_resident_greedy_session,
    flatten_explicit_state,
    unflatten_explicit_state,
)


def test_probe_s1_outputs_accumulation() -> None:
    """S1 — ``outputs=`` tracks simple mutable array state across steps."""
    pytest.xfail("P1 outputs= state threading is not reliable in the current MLX environment")
    state = [mx.zeros((1, 4), dtype=mx.float32)]

    @partial(mx.compile, outputs=state)
    def compiled_accumulate(x: mx.array) -> mx.array:
        state[0] = state[0] + x
        return state[0]

    for step in range(8):
        result = compiled_accumulate(mx.ones((1, 4), dtype=mx.float32))
        mx.eval(result)
        expected = step + 1
        actual = float(result[0, 0].item())
        assert abs(actual - expected) < 1e-5, f"step {step}: expected {expected}, got {actual}"


def test_probe_s2_slice_write_tracking() -> None:
    """S2 — ``outputs=`` tracks in-place slice writes on a pre-allocated buffer."""
    buf = [mx.zeros((1, 1, 512, 64), dtype=mx.float16)]

    @partial(mx.compile, outputs=buf)
    def compiled_cache_write(new_kv: mx.array, offset: int) -> mx.array:
        buf[0][..., offset : offset + 1, :] = new_kv
        return buf[0][..., : offset + 1, :]

    for step in range(16):
        new_kv = mx.ones((1, 1, 1, 64), dtype=mx.float16) * step
        result = compiled_cache_write(new_kv, step)
        mx.eval(result)
        slot_val = float(result[0, 0, step, 0].item())
        assert abs(slot_val - step) < 0.1, f"step {step}: slot={slot_val}, expected {step}"


def test_probe_s3_preallocation_prevents_grow(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 — pre-allocation must prevent ``_grow()`` during the decode loop."""
    prefill_len = 256
    max_tokens = 128
    total_slots = prefill_len + max_tokens
    cache = KVCache()

    seed_k = mx.zeros((1, 8, prefill_len, 64), dtype=mx.float16)
    seed_v = mx.zeros((1, 8, prefill_len, 64), dtype=mx.float16)
    cache.update_and_fetch(seed_k, seed_v)
    cache.ensure_total_slots(total_slots)
    cache.offset = prefill_len

    grow_calls: list[int] = []
    original_grow = KVCache._grow

    def patched_grow(
        self: KVCache,
        keys: mx.array,
        values: mx.array,
        prev: int,
        n_new: int,
    ) -> None:
        grow_calls.append(1)
        original_grow(self, keys, values, prev, n_new)

    monkeypatch.setattr(KVCache, "_grow", patched_grow)
    for _ in range(max_tokens):
        new_k = mx.ones((1, 8, 1, 64), dtype=mx.float16)
        new_v = mx.ones((1, 8, 1, 64), dtype=mx.float16)
        cache.update_and_fetch(new_k, new_v)

    assert len(grow_calls) == 0, f"_grow called {len(grow_calls)} times"


def test_probe_s4_full_stateful_preallocated_kvcache() -> None:
    """S4 — full stateful probe with pre-allocated ``KVCache`` arrays."""
    pytest.xfail("P1 full stateful KVCache outputs= probe is retired after the P1 stop decision")
    n_layers = 2
    head_dim = 64
    n_kv_heads = 4
    total_slots = 300

    caches = [KVCache() for _ in range(n_layers)]
    for cache in caches:
        seed_k = mx.zeros((1, n_kv_heads, 1, head_dim), dtype=mx.float16)
        seed_v = mx.zeros((1, n_kv_heads, 1, head_dim), dtype=mx.float16)
        cache.update_and_fetch(seed_k, seed_v)
        cache.offset = 0
        cache.ensure_total_slots(total_slots)

    cache_arrays = [arr for cache in caches for arr in cache.tracked_arrays]
    step_counter = [0]

    @partial(mx.compile, outputs=cache_arrays)
    def compiled_mock_step(token: mx.array) -> mx.array:
        offset = step_counter[0]
        new_k = mx.broadcast_to(token.reshape(1, 1, 1, 1), (1, n_kv_heads, 1, head_dim)).astype(
            mx.float16
        )
        new_v = new_k
        for cache in caches:
            cache.tracked_arrays[0][..., offset : offset + 1, :] = new_k
            cache.tracked_arrays[1][..., offset : offset + 1, :] = new_v
        return cast(mx.array, caches[0].tracked_arrays[0][..., : offset + 1, :])

    for step in range(280):
        token = mx.array([step % 1000], dtype=mx.int32)
        result = compiled_mock_step(token)
        step_counter[0] += 1
        mx.eval(result)
        if step in {0, 1, 10, 255, 256, 279}:
            slot_val = float(result[0, 0, step, 0].item())
            expected = float(step % 1000)
            assert abs(slot_val - expected) < 0.5, (
                f"step {step}: slot value={slot_val}, expected~{expected}"
            )


def test_explicit_state_flatten_roundtrip() -> None:
    """Experimental P2 helpers preserve layer ordering and array identities."""
    state = flatten_explicit_state(
        (
            mx.zeros((1, 2, 4, 3), dtype=mx.float16),
            mx.ones((1, 2, 4, 3), dtype=mx.float16),
        ),
        (
            mx.full((1, 2, 4, 3), 2.0, dtype=mx.float16),
            mx.full((1, 2, 4, 3), 3.0, dtype=mx.float16),
        ),
        (
            mx.array([4], dtype=mx.int32),
            mx.array([7], dtype=mx.int32),
        ),
    )

    k_arrays, v_arrays, offsets = unflatten_explicit_state(state, num_layers=2)
    assert len(k_arrays) == 2
    assert len(v_arrays) == 2
    assert len(offsets) == 2
    assert int(offsets[0].item()) == 4
    assert int(offsets[1].item()) == 7
    assert float(v_arrays[1][0, 0, 0, 0].item()) == pytest.approx(3.0)


def test_explicit_state_export_materialize_roundtrip() -> None:
    """Experimental P2 can roundtrip eager ``KVCache`` state at the boundary."""
    cache = KVCache()
    keys = mx.arange(24, dtype=mx.float16).reshape(1, 2, 4, 3)
    values = (mx.arange(24, dtype=mx.float16) + 100).reshape(1, 2, 4, 3)
    cache.update_and_fetch(keys, values)

    state = export_kvcache_to_explicit_state([cache], total_slots=8)
    rebuilt = materialize_explicit_state_to_kvcache(state, num_layers=1)
    rebuilt_state = rebuilt[0].state
    assert rebuilt_state is not None
    rebuilt_keys, rebuilt_values = rebuilt_state
    assert rebuilt_keys.shape == keys.shape
    assert rebuilt_values.shape == values.shape
    assert mx.allclose(rebuilt_keys, keys, atol=0).item()
    assert mx.allclose(rebuilt_values, values, atol=0).item()


def test_explicit_state_step_fn_returns_logprobs_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def __call__(
            self,
            input_ids: mx.array,
            *,
            cache: list[Any] | None = None,
            mask: Any = None,
            input_embeddings: mx.array | None = None,
        ) -> mx.array:
            del cache, mask, input_embeddings
            row = mx.array([[5.0, 1.0, -2.0]], dtype=mx.float32)
            return mx.broadcast_to(row, (input_ids.shape[0], input_ids.shape[1], 3))

    sampler_calls = {"count": 0}

    def _sampler(logprobs: mx.array) -> mx.array:
        sampler_calls["count"] += 1
        return mx.argmax(logprobs, axis=-1)

    monkeypatch.setattr(
        "mlxs.generate.recipe.mx.compile",
        lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda inner: inner),
    )

    recipe = Recipe(
        sampler=_sampler,
        logits_processors=(),
        has_processors=False,
        processor_context_size=0,
        emit_logprobs=False,
        emit_top_logprobs=False,
        top_logprobs_k=0,
        compile_mode=CompileMode.ON,
    )
    initial_state = flatten_explicit_state(
        (mx.zeros((1, 1, 4, 2), dtype=mx.float16),),
        (mx.zeros((1, 1, 4, 2), dtype=mx.float16),),
        (mx.array([0], dtype=mx.int32),),
    )
    step_fn, snapshot_state = build_explicit_state_step_fn(
        cast(Any, _Model()),
        recipe,
        _generation_stream,
        num_layers=1,
        initial_state=initial_state,
    )
    outputs = step_fn(mx.array([1], dtype=mx.int32), mx.array([0], dtype=mx.int32))
    logprobs, *offsets = outputs
    mx.eval(logprobs, *offsets)

    assert sampler_calls["count"] == 0
    assert logprobs.shape == (3,)
    assert len(offsets) == 1
    rebuilt_state = snapshot_state(cast(tuple[mx.array, ...], tuple(offsets)))
    assert len(rebuilt_state) == 3


def test_explicit_state_step_fn_returns_raw_logits_when_processors_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def __call__(
            self,
            input_ids: mx.array,
            *,
            cache: list[Any] | None = None,
            mask: Any = None,
            input_embeddings: mx.array | None = None,
        ) -> mx.array:
            del cache, mask, input_embeddings
            row = mx.array([[5.0, 1.0, -2.0]], dtype=mx.float32)
            return mx.broadcast_to(row, (input_ids.shape[0], input_ids.shape[1], 3))

    sampler_calls = {"count": 0}

    def _sampler(logprobs: mx.array) -> mx.array:
        sampler_calls["count"] += 1
        return mx.argmax(logprobs, axis=-1)

    monkeypatch.setattr(
        "mlxs.generate.recipe.mx.compile",
        lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda inner: inner),
    )

    recipe = Recipe(
        sampler=_sampler,
        logits_processors=(object(),),
        has_processors=True,
        processor_context_size=20,
        emit_logprobs=False,
        emit_top_logprobs=False,
        top_logprobs_k=0,
        compile_mode=CompileMode.ON,
    )
    initial_state = flatten_explicit_state(
        (mx.zeros((1, 1, 4, 2), dtype=mx.float16),),
        (mx.zeros((1, 1, 4, 2), dtype=mx.float16),),
        (mx.array([0], dtype=mx.int32),),
    )
    step_fn, snapshot_state = build_explicit_state_step_fn(
        cast(Any, _Model()),
        recipe,
        _generation_stream,
        num_layers=1,
        initial_state=initial_state,
    )
    outputs = step_fn(mx.array([1], dtype=mx.int32), mx.array([0], dtype=mx.int32))
    logits, *offsets = outputs
    mx.eval(logits, *offsets)

    assert sampler_calls["count"] == 0
    assert logits.shape == (3,)
    assert float(logits[0].item()) == pytest.approx(5.0)
    assert len(offsets) == 1
    rebuilt_state = snapshot_state(cast(tuple[mx.array, ...], tuple(offsets)))
    assert len(rebuilt_state) == 3


def test_resident_greedy_session_advances_without_host_threaded_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CachingModel:
        def __call__(
            self,
            input_ids: mx.array,
            *,
            cache: list[Any] | None = None,
            mask: Any = None,
            input_embeddings: mx.array | None = None,
        ) -> mx.array:
            del mask, input_embeddings
            assert cache is not None
            token_id = input_ids[:, -1]
            kv = mx.broadcast_to(token_id.reshape(1, 1, 1, 1), (1, 1, 1, 2)).astype(mx.float16)
            cache[0].update_and_fetch(kv, kv + 10)
            next_id = (token_id[0] + 1) % 8
            row = mx.where(
                mx.arange(8) == next_id,
                mx.array(5.0, dtype=mx.float32),
                -mx.arange(8, dtype=mx.float32) - 5.0,
            )
            return row.reshape(1, 1, 8)

    monkeypatch.setattr(
        "mlxs.generate.recipe.mx.compile",
        lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda inner: inner),
    )

    recipe = Recipe(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        logits_processors=(),
        has_processors=False,
        processor_context_size=0,
        emit_logprobs=False,
        emit_top_logprobs=False,
        top_logprobs_k=0,
        compile_mode=CompileMode.ON,
    )
    initial_state = flatten_explicit_state(
        (mx.zeros((1, 1, 4, 2), dtype=mx.float16),),
        (mx.zeros((1, 1, 4, 2), dtype=mx.float16),),
        (mx.array([0], dtype=mx.int32),),
    )

    session = build_resident_greedy_session(
        cast(Any, _CachingModel()),
        recipe,
        _generation_stream,
        num_layers=1,
        initial_state=initial_state,
        seed_token=mx.array([1], dtype=mx.int32),
    )

    token_a = session.advance()
    token_b = session.advance()
    mx.eval(token_a, token_b)

    assert int(token_a.item()) == 2
    assert int(token_b.item()) == 3

    rebuilt = materialize_explicit_state_to_kvcache(session.snapshot_state(), num_layers=1)
    rebuilt_state = rebuilt[0].state
    assert rebuilt_state is not None
    rebuilt_keys, rebuilt_values = rebuilt_state
    assert rebuilt_keys.shape[2] == 2
    assert float(rebuilt_keys[0, 0, 0, 0].item()) == pytest.approx(1.0)
    assert float(rebuilt_keys[0, 0, 1, 0].item()) == pytest.approx(2.0)
    assert float(rebuilt_values[0, 0, 0, 0].item()) == pytest.approx(11.0)
    assert float(rebuilt_values[0, 0, 1, 0].item()) == pytest.approx(12.0)


def test_resident_state_decode_experiment_matches_explicit_state_greedy() -> None:
    class _CachingModel:
        def __init__(self) -> None:
            self._cache_template = [KVCache()]

        def __call__(
            self,
            input_ids: mx.array,
            *,
            cache: list[KVCache] | None = None,
            mask: Any = None,
            input_embeddings: mx.array | None = None,
        ) -> mx.array:
            del mask, input_embeddings
            assert cache is not None
            token_id = input_ids[:, -1]
            kv = mx.broadcast_to(token_id.reshape(1, 1, 1, 1), (1, 1, 1, 2)).astype(mx.float16)
            cache[0].update_and_fetch(kv, kv + 10)
            next_id = (token_id[0] + 1) % 8
            row = mx.where(
                mx.arange(8) == next_id,
                mx.array(5.0, dtype=mx.float32),
                -mx.arange(8, dtype=mx.float32) - 5.0,
            )
            return row.reshape(1, 1, 8)

        def make_cache(self) -> list[KVCache]:
            return [KVCache()]

    recipe = Recipe(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        logits_processors=(),
        has_processors=False,
        processor_context_size=0,
        emit_logprobs=False,
        emit_top_logprobs=False,
        top_logprobs_k=0,
        compile_mode=CompileMode.ON,
    )
    prompt_tokens = [1]

    explicit_tokens, explicit_cache = run_explicit_state_decode_experiment(
        cast(Any, _CachingModel()),
        prompt_tokens,
        recipe,
        max_tokens=4,
        prefill_step_size=8,
        stream=_generation_stream,
    )
    resident_tokens, resident_cache = run_resident_state_decode_experiment(
        cast(Any, _CachingModel()),
        prompt_tokens,
        recipe,
        max_tokens=4,
        prefill_step_size=8,
        stream=_generation_stream,
    )

    assert resident_tokens == explicit_tokens
    resident_state = resident_cache[0].state
    explicit_state = explicit_cache[0].state
    assert resident_state is not None
    assert explicit_state is not None
    assert mx.allclose(resident_state[0], explicit_state[0], atol=0).item()
    assert mx.allclose(resident_state[1], explicit_state[1], atol=0).item()


def test_resident_state_decode_experiment_honors_eos_stop() -> None:
    class _EosModel:
        def __call__(
            self,
            input_ids: mx.array,
            *,
            cache: list[KVCache] | None = None,
            mask: Any = None,
            input_embeddings: mx.array | None = None,
        ) -> mx.array:
            del mask, input_embeddings
            assert cache is not None
            token_id = input_ids[:, -1]
            kv = mx.broadcast_to(token_id.reshape(1, 1, 1, 1), (1, 1, 1, 2)).astype(mx.float16)
            cache[0].update_and_fetch(kv, kv)
            row = mx.where(
                mx.arange(8) == 0,
                mx.array(5.0, dtype=mx.float32),
                -mx.arange(8, dtype=mx.float32) - 5.0,
            )
            return row.reshape(1, 1, 8)

        def make_cache(self) -> list[KVCache]:
            return [KVCache()]

    recipe = Recipe(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        logits_processors=(),
        has_processors=False,
        processor_context_size=0,
        emit_logprobs=False,
        emit_top_logprobs=False,
        top_logprobs_k=0,
        compile_mode=CompileMode.ON,
    )

    tokens, rebuilt_cache = run_resident_state_decode_experiment(
        cast(Any, _EosModel()),
        [1],
        recipe,
        max_tokens=8,
        eos_token_id=0,
        prefill_step_size=8,
        stream=_generation_stream,
    )

    assert tokens == [0]
    assert rebuilt_cache[0].state is not None
