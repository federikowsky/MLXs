"""Decode Engine V4 recipe specialization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache

SamplerFn = Callable[[mx.array], mx.array]
StepFn = Callable[[mx.array], tuple[mx.array, mx.array]]
LogitsStepFn = Callable[[mx.array], mx.array]
ExplicitState = tuple[mx.array, ...]
ExplicitOffsets = tuple[mx.array, ...]
ExplicitStateStepFn = Callable[..., tuple[mx.array, ...]]
ExplicitStateSnapshotFn = Callable[[ExplicitOffsets], ExplicitState]
ResidentGreedyAdvanceFn = Callable[[], mx.array]
ResidentGreedySnapshotFn = Callable[[], ExplicitState]


class CompileMode(StrEnum):
    """Compile mode for a recipe."""

    OFF = "off"
    ON = "on"


@dataclass(frozen=True, slots=True)
class Recipe:
    """Frozen per-request specialization."""

    sampler: SamplerFn
    logits_processors: tuple[object, ...]
    has_processors: bool
    processor_context_size: int
    emit_logprobs: bool
    emit_top_logprobs: bool
    top_logprobs_k: int
    compile_mode: CompileMode


@dataclass(frozen=True, slots=True)
class ResidentGreedySession:
    """Minimal resident-state greedy session for Gate 1 / Gate 2 probing."""

    advance: ResidentGreedyAdvanceFn
    snapshot_state: ResidentGreedySnapshotFn


class _ExplicitStateLayerCache:
    """Experimental P2 cache adapter backed only by explicit array state."""

    __slots__ = ("_keys", "_offset", "_values")

    def __init__(self, keys: mx.array, values: mx.array, offset: mx.array) -> None:
        self._keys = keys
        self._values = values
        self._offset = offset

    @property
    def offset(self) -> mx.array:
        return self._offset

    @property
    def keys_array(self) -> mx.array:
        return self._keys

    @property
    def values_array(self) -> mx.array:
        return self._values

    @property
    def offset_array(self) -> mx.array:
        return self._offset

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if keys.shape[2] != 1 or values.shape[2] != 1:
            raise NotImplementedError(
                "Experimental P2 explicit-state cache supports single-token decode only"
            )

        self._keys = mx.slice_update(
            self._keys,
            keys,
            start_indices=self._offset,
            axes=(2,),
        )
        self._values = mx.slice_update(
            self._values,
            values,
            start_indices=self._offset,
            axes=(2,),
        )
        self._offset = self._offset + 1
        return self._keys, self._values

    def make_mask(
        self,
        n: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> mx.array:
        del return_array
        if n != 1:
            raise NotImplementedError(
                "Experimental P2 explicit-state cache supports single-token decode masks only"
            )
        if window_size is not None:
            raise NotImplementedError(
                "Experimental P2 explicit-state cache does not support sliding-window masks"
            )

        positions = mx.arange(self._keys.shape[2], dtype=self._offset.dtype).reshape(1, 1, 1, -1)
        return positions < (self._offset + n).reshape(1, 1, 1, 1)


def flatten_explicit_state(
    k_arrays: tuple[mx.array, ...],
    v_arrays: tuple[mx.array, ...],
    offsets: tuple[mx.array, ...],
) -> ExplicitState:
    """Flatten explicit per-layer state into a compile-friendly tuple."""
    flat: list[mx.array] = []
    for k_array, v_array, offset in zip(k_arrays, v_arrays, offsets, strict=True):
        flat.extend((k_array, v_array, offset))
    return tuple(flat)


def unflatten_explicit_state(
    state_arrays: ExplicitState,
    *,
    num_layers: int,
) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...], tuple[mx.array, ...]]:
    """Restore explicit per-layer state from a flat tuple."""
    expected = num_layers * 3
    if len(state_arrays) != expected:
        raise ValueError(
            f"Expected {expected} explicit state arrays for {num_layers} layers, "
            f"got {len(state_arrays)}"
        )

    k_arrays: list[mx.array] = []
    v_arrays: list[mx.array] = []
    offsets: list[mx.array] = []
    for idx in range(num_layers):
        base = idx * 3
        k_arrays.append(state_arrays[base])
        v_arrays.append(state_arrays[base + 1])
        offsets.append(state_arrays[base + 2])
    return tuple(k_arrays), tuple(v_arrays), tuple(offsets)


def build_step_fn(
    model: nn.Module,
    cache: list[KVCache],
    recipe: Recipe,
    stream: mx.Stream,
) -> StepFn:
    """Build the specialized step closure for the active recipe."""
    if recipe.has_processors or recipe.logits_processors:
        raise NotImplementedError(
            "Processor-aware decode uses build_logits_step_fn() instead"
        )

    sampler = recipe.sampler

    if recipe.compile_mode is CompileMode.ON:

        @mx.compile
        def compiled_step(prev_token: mx.array) -> tuple[mx.array, mx.array]:
            with mx.stream(stream):
                logits = cast(mx.array, model(prev_token[None], cache=cache))
                logits = logits[:, -1, :]
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                next_token = sampler(logprobs)
                return next_token, logprobs.squeeze(0)

        return cast(StepFn, compiled_step)

    def step_fn(prev_token: mx.array) -> tuple[mx.array, mx.array]:
        with mx.stream(stream):
            logits = cast(mx.array, model(prev_token[None], cache=cache))
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            next_token = sampler(logprobs)
            return next_token, logprobs.squeeze(0)

    return step_fn


def build_logits_step_fn(
    model: nn.Module,
    cache: list[KVCache],
    recipe: Recipe,
    stream: mx.Stream,
) -> LogitsStepFn:
    """Build a logits-only step closure for processor-aware decode."""
    if recipe.compile_mode is not CompileMode.OFF:
        raise NotImplementedError("build_logits_step_fn() is only used by eager decode")

    def step_fn(prev_token: mx.array) -> mx.array:
        with mx.stream(stream):
            logits = cast(mx.array, model(prev_token[None], cache=cache))
            logits = logits[:, -1, :]
            return logits.squeeze(0)

    return step_fn


def build_explicit_state_step_fn(
    model: nn.Module,
    recipe: Recipe,
    stream: mx.Stream,
    *,
    num_layers: int,
    initial_state: ExplicitState,
) -> tuple[ExplicitStateStepFn, ExplicitStateSnapshotFn]:
    """Build the experimental P2 step closure with explicit array state."""
    if recipe.compile_mode is not CompileMode.ON:
        raise NotImplementedError("Experimental P2 explicit-state decode requires compile mode on")
    k_arrays, v_arrays, _ = unflatten_explicit_state(initial_state, num_layers=num_layers)
    tracked_state: list[mx.array] = [*k_arrays, *v_arrays]

    @partial(mx.compile, inputs=tracked_state, outputs=tracked_state)
    def compiled_step(prev_token: mx.array, *state_arrays: mx.array) -> tuple[mx.array, ...]:
        if len(state_arrays) != num_layers:
            raise ValueError(
                f"Expected {num_layers} explicit offsets for {num_layers} layers, "
                f"got {len(state_arrays)}"
            )
        offsets = cast(ExplicitOffsets, tuple(state_arrays))
        caches = [
            _ExplicitStateLayerCache(
                tracked_state[idx],
                tracked_state[num_layers + idx],
                offset,
            )
            for idx, offset in enumerate(offsets)
        ]
        with mx.stream(stream):
            logits = cast(mx.array, model(prev_token[None], cache=caches))
            logits = logits[:, -1, :]

        new_offsets: list[mx.array] = []
        for idx, cache in enumerate(caches):
            tracked_state[idx] = cache.keys_array
            tracked_state[num_layers + idx] = cache.values_array
            new_offsets.append(cache.offset_array)
        if recipe.has_processors or recipe.logits_processors:
            return (logits.squeeze(0), *new_offsets)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return (logprobs.squeeze(0), *new_offsets)

    def snapshot_state(offsets: ExplicitOffsets) -> ExplicitState:
        if len(offsets) != num_layers:
            raise ValueError(
                f"Expected {num_layers} explicit offsets for {num_layers} layers, "
                f"got {len(offsets)}"
            )
        return flatten_explicit_state(
            tuple(tracked_state[idx] for idx in range(num_layers)),
            tuple(tracked_state[num_layers + idx] for idx in range(num_layers)),
            offsets,
        )

    return cast(ExplicitStateStepFn, compiled_step), snapshot_state


def build_resident_greedy_session(
    model: nn.Module,
    recipe: Recipe,
    stream: mx.Stream,
    *,
    num_layers: int,
    initial_state: ExplicitState,
    seed_token: mx.array,
) -> ResidentGreedySession:
    """Build a resident greedy session with no host-threaded live continuation state."""
    if recipe.compile_mode is not CompileMode.ON:
        raise NotImplementedError("Resident-state session requires compile mode on")
    if recipe.has_processors or recipe.logits_processors:
        raise NotImplementedError("Resident-state session currently supports greedy decode only")
    if recipe.emit_logprobs or recipe.emit_top_logprobs:
        raise NotImplementedError("Resident-state session currently does not emit logprobs")

    k_arrays, v_arrays, offsets = unflatten_explicit_state(initial_state, num_layers=num_layers)
    tracked_state: list[mx.array] = [
        *k_arrays,
        *v_arrays,
        *offsets,
        seed_token.astype(mx.int32).reshape(1),
    ]
    offset_base = num_layers * 2
    token_index = num_layers * 3

    @partial(mx.compile, inputs=tracked_state, outputs=tracked_state)
    def advance() -> mx.array:
        caches = [
            _ExplicitStateLayerCache(
                tracked_state[idx],
                tracked_state[num_layers + idx],
                tracked_state[offset_base + idx],
            )
            for idx in range(num_layers)
        ]
        prev_token = tracked_state[token_index]
        with mx.stream(stream):
            logits = cast(mx.array, model(prev_token[None], cache=caches))
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            next_token = recipe.sampler(logprobs).astype(mx.int32).reshape(1)

        for idx, cache in enumerate(caches):
            tracked_state[idx] = cache.keys_array
            tracked_state[num_layers + idx] = cache.values_array
            tracked_state[offset_base + idx] = cache.offset_array
        tracked_state[token_index] = next_token
        return next_token

    def snapshot_state() -> ExplicitState:
        return flatten_explicit_state(
            tuple(tracked_state[idx] for idx in range(num_layers)),
            tuple(tracked_state[num_layers + idx] for idx in range(num_layers)),
            tuple(tracked_state[offset_base + idx] for idx in range(num_layers)),
        )

    return ResidentGreedySession(advance=advance, snapshot_state=snapshot_state)


__all__ = [
    "CompileMode",
    "ExplicitOffsets",
    "ExplicitState",
    "ExplicitStateSnapshotFn",
    "ExplicitStateStepFn",
    "LogitsStepFn",
    "Recipe",
    "ResidentGreedyAdvanceFn",
    "ResidentGreedySession",
    "ResidentGreedySnapshotFn",
    "SamplerFn",
    "StepFn",
    "build_explicit_state_step_fn",
    "build_logits_step_fn",
    "build_resident_greedy_session",
    "build_step_fn",
    "flatten_explicit_state",
    "unflatten_explicit_state",
]
