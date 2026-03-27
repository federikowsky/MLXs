"""Deterministic scenario builders for adaptive KV benchmarks.

All scenarios are single-request: one prompt (token ids) + one generate() call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlxs._types import GenerateOptions

SCENARIO_NAMES: tuple[str, ...] = (
    "smoke",
    "long_static",
    "topic_drift",
    "delayed_topic_return",
    "oscillation",
    "early_constraint_retention",
    "hard_pressure_context",
)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Built prompt and generation options for one scenario."""

    name: str
    prompt_token_ids: list[int]
    options: GenerateOptions
    description: str
    meta: dict[str, Any]


def _repeat_to_min_tokens(tokenizer: Any, fragment: str, min_tokens: int) -> list[int]:
    if min_tokens <= 0:
        return []
    ids: list[int] = []
    frag_ids = tokenizer.encode(fragment)
    if not frag_ids:
        raise ValueError("scenario fragment encodes to empty token list")
    while len(ids) < min_tokens:
        ids.extend(frag_ids)
    return ids[:min_tokens]


def build_scenario(
    name: str,
    tokenizer: Any,
    *,
    seed: int | None = 42,
    prompt_target_tokens: int = 512,
    decode_tokens: int = 64,
) -> ScenarioResult:
    """Build (prompt_ids, GenerateOptions) for a named scenario."""
    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {name!r}; one of {SCENARIO_NAMES}")

    static = "The quick brown fox jumps over the lazy dog. "
    topic_a = "Topic Alpha: astronomy, nebulae, and stellar formation. "
    topic_b = "Topic Beta: linear algebra, norms, and orthogonal maps. "
    constraint = (
        "SYSTEM CONSTRAINT: You must remember code ALPHA-774 for the final answer. "
        "Do not contradict this. "
    )

    if name == "smoke":
        p = _repeat_to_min_tokens(tokenizer, static, min(32, prompt_target_tokens))
        opts = GenerateOptions(max_tokens=min(8, decode_tokens), temperature=0.0, seed=seed)
        return ScenarioResult(
            name=name,
            prompt_token_ids=p,
            options=opts,
            description="Minimal prompt for sanity / tiny models.",
            meta={"prompt_tokens": len(p)},
        )

    if name == "long_static":
        p = _repeat_to_min_tokens(tokenizer, static, prompt_target_tokens)
        opts = GenerateOptions(max_tokens=decode_tokens, temperature=0.0, seed=seed)
        return ScenarioResult(
            name=name,
            prompt_token_ids=p,
            options=opts,
            description="Long homogeneous context + decode.",
            meta={"prompt_tokens": len(p)},
        )

    if name == "topic_drift":
        a_len = max(prompt_target_tokens // 3, 64)
        b_len = prompt_target_tokens - a_len
        p = _repeat_to_min_tokens(tokenizer, topic_a, a_len)
        p.extend(_repeat_to_min_tokens(tokenizer, topic_b, b_len))
        opts = GenerateOptions(max_tokens=decode_tokens, temperature=0.0, seed=seed)
        return ScenarioResult(
            name=name,
            prompt_token_ids=p,
            options=opts,
            description="Long topic A then long topic B; exercises aging on A.",
            meta={"prompt_tokens": len(p), "a_tokens": a_len, "b_tokens": b_len},
        )

    if name == "delayed_topic_return":
        third = max(prompt_target_tokens // 3, 48)
        p = _repeat_to_min_tokens(tokenizer, topic_a, third)
        p.extend(_repeat_to_min_tokens(tokenizer, topic_b, third))
        p.extend(_repeat_to_min_tokens(tokenizer, topic_a, prompt_target_tokens - len(p)))
        opts = GenerateOptions(max_tokens=decode_tokens, temperature=0.0, seed=seed)
        return ScenarioResult(
            name=name,
            prompt_token_ids=p,
            options=opts,
            description="A, then B, then A again; may interact with promotion/recovery.",
            meta={"prompt_tokens": len(p)},
        )

    if name == "oscillation":
        chunk = max(16, prompt_target_tokens // 20)
        p = []
        while len(p) < prompt_target_tokens:
            need = min(chunk, prompt_target_tokens - len(p))
            p.extend(_repeat_to_min_tokens(tokenizer, topic_a, need))
            if len(p) >= prompt_target_tokens:
                break
            need = min(chunk, prompt_target_tokens - len(p))
            p.extend(_repeat_to_min_tokens(tokenizer, topic_b, need))
        opts = GenerateOptions(max_tokens=decode_tokens, temperature=0.0, seed=seed)
        return ScenarioResult(
            name=name,
            prompt_token_ids=p,
            options=opts,
            description="Interleaved A/B chunks; stress policy stability.",
            meta={"prompt_tokens": len(p), "chunk_tokens": chunk},
        )

    if name == "early_constraint_retention":
        c_ids = tokenizer.encode(constraint)
        noise_len = max(prompt_target_tokens - len(c_ids), 64)
        noise = _repeat_to_min_tokens(tokenizer, static, noise_len)
        p = list(c_ids) + noise
        if len(p) > prompt_target_tokens:
            p = p[:prompt_target_tokens]
        opts = GenerateOptions(max_tokens=decode_tokens, temperature=0.0, seed=seed)
        return ScenarioResult(
            name=name,
            prompt_token_ids=p,
            options=opts,
            description="Strong early instruction + long filler + decode.",
            meta={"prompt_tokens": len(p), "constraint_tokens": len(c_ids)},
        )

    if name == "hard_pressure_context":
        p = _repeat_to_min_tokens(tokenizer, static, max(prompt_target_tokens, 256))
        opts = GenerateOptions(max_tokens=max(decode_tokens, 48), temperature=0.0, seed=seed)
        return ScenarioResult(
            name=name,
            prompt_token_ids=p,
            options=opts,
            description="Sized for eviction/recovery under hard budgets.",
            meta={"prompt_tokens": len(p)},
        )

    raise AssertionError(name)
