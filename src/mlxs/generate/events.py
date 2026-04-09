"""Decode Engine V4 caller-wrapper event stream."""

from __future__ import annotations

import time
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, cast

import mlx.core as mx

from mlxs._types import FinishReason, TokenEvent, TokenLogprobs, TopLogprob
from mlxs.generate.recipe import Recipe
from mlxs.generate.runtime import DecodePlan

if TYPE_CHECKING:
    from mlxs.generate.profile import DecodeProfiler


def token_event_stream(
    core_iter: Generator[tuple[int, object], None, None],
    plan: DecodePlan,
    recipe: Recipe,
    max_tokens: int,
    *,
    profiler: DecodeProfiler | None = None,
) -> Iterator[TokenEvent]:
    """Convert raw core tuples into public ``TokenEvent`` values."""
    if profiler is None:
        yield from _token_event_stream(core_iter, plan, recipe, max_tokens)
        return
    yield from _profiled_token_event_stream(core_iter, plan, recipe, max_tokens, profiler)


def _token_event_stream(
    core_iter: Generator[tuple[int, object], None, None],
    plan: DecodePlan,
    recipe: Recipe,
    max_tokens: int,
) -> Iterator[TokenEvent]:
    n_emitted = 0
    try:
        while True:
            token_id, logprobs_lazy = next(core_iter)
            text = plan.decoder(token_id)
            finish_reason = plan.stop.check(token_id, text)
            if finish_reason is None and n_emitted + 1 == max_tokens:
                finish_reason = FinishReason.LENGTH

            event = TokenEvent(
                token_id=token_id,
                text=text,
                finish_reason=finish_reason,
                logprobs=_materialize_logprobs(
                    token_id,
                    logprobs_lazy,
                    plan,
                    recipe,
                ),
                prompt_tokens=plan.prompt_token_count,
                generation_tokens=plan.stop.generated_count,
            )
            yield event
            n_emitted += 1

            if finish_reason is not None:
                core_iter.close()
                return
    except StopIteration:
        return
    finally:
        core_iter.close()


def _profiled_token_event_stream(
    core_iter: Generator[tuple[int, object], None, None],
    plan: DecodePlan,
    recipe: Recipe,
    max_tokens: int,
    profiler: DecodeProfiler,
) -> Iterator[TokenEvent]:
    n_emitted = 0
    try:
        while True:
            token_id, logprobs_lazy = next(core_iter)

            t0 = time.perf_counter()
            text = plan.decoder(token_id)
            profiler.tokenizer_wall_s.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            finish_reason = plan.stop.check(token_id, text)
            profiler.stop_check_wall_s.append(time.perf_counter() - t0)
            if finish_reason is None and n_emitted + 1 == max_tokens:
                finish_reason = FinishReason.LENGTH

            t0 = time.perf_counter()
            event = TokenEvent(
                token_id=token_id,
                text=text,
                finish_reason=finish_reason,
                logprobs=_materialize_logprobs(
                    token_id,
                    logprobs_lazy,
                    plan,
                    recipe,
                ),
                prompt_tokens=plan.prompt_token_count,
                generation_tokens=plan.stop.generated_count,
            )
            profiler.event_build_wall_s.append(time.perf_counter() - t0)

            yield event
            n_emitted += 1

            if finish_reason is not None:
                core_iter.close()
                return
    except StopIteration:
        return
    finally:
        core_iter.close()


def _materialize_logprobs(
    token_id: int,
    logprobs_lazy: object,
    plan: DecodePlan,
    recipe: Recipe,
) -> TokenLogprobs | None:
    if not recipe.emit_logprobs:
        return None
    if logprobs_lazy is None:
        return None
    logprobs = cast(mx.array, logprobs_lazy)

    mx.eval(logprobs)
    token_logprob = float(logprobs[token_id].item())

    top_logprobs: tuple[TopLogprob, ...] = ()
    if recipe.emit_top_logprobs and recipe.top_logprobs_k > 0:
        k = min(recipe.top_logprobs_k, int(logprobs.shape[0]))
        top_indices = mx.argsort(-logprobs)[:k]
        mx.eval(top_indices)
        top_logprobs = tuple(
            TopLogprob(
                token_id=idx,
                token=plan.decoder(idx),
                logprob=float(logprobs[idx].item()),
            )
            for idx in (int(top_idx.item()) for top_idx in top_indices)
        )

    return TokenLogprobs(
        token_logprob=token_logprob,
        top_logprobs=top_logprobs,
    )


__all__ = ["token_event_stream"]
