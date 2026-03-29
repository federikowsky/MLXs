"""Named benchmark scenarios — MLXs vs mlx-lm (no runtime changes in core)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """One reproducible generation setting for both backends where supported."""

    id: str
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    logprobs: bool = False
    top_logprobs: int = 0
    #: MLXs: passed to ``generate(..., compile_decode=...)``. Ignored by mlx-lm.
    compile_decode: bool = True
    quantized_kv_start: int = 0
    kv_bits: int | None = None
    kv_group_size: int = 64
    #: mlx-lm always builds a full-vocab logprob vector; optionally ``mx.eval`` it per step
    #: to approximate MLXs logprob payload materialization cost.
    materialize_logprobs_mlx_lm: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


# Short / long decode lengths (tokenizer-agnostic targets; actual stop = max_tokens).
_SHORT = 128
_LONG = 512
_SAMPLING_TEMP = 0.8
_SAMPLING_TOP_P = 0.95
_REP_PENALTY = 1.12

BUILTIN_SCENARIOS: dict[str, ScenarioSpec] = {
    "greedy_short": ScenarioSpec(
        id="greedy_short",
        max_tokens=_SHORT,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        compile_decode=True,
    ),
    "greedy_long": ScenarioSpec(
        id="greedy_long",
        max_tokens=_LONG,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        compile_decode=True,
    ),
    "sampling_short": ScenarioSpec(
        id="sampling_short",
        max_tokens=_SHORT,
        temperature=_SAMPLING_TEMP,
        top_p=_SAMPLING_TOP_P,
        top_k=0,
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        compile_decode=True,
    ),
    "sampling_long": ScenarioSpec(
        id="sampling_long",
        max_tokens=_LONG,
        temperature=_SAMPLING_TEMP,
        top_p=_SAMPLING_TOP_P,
        top_k=0,
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        compile_decode=True,
    ),
    "repetition_penalty": ScenarioSpec(
        id="repetition_penalty",
        max_tokens=_SHORT,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=_REP_PENALTY,
        logprobs=False,
        top_logprobs=0,
        compile_decode=True,
    ),
    "logprobs": ScenarioSpec(
        id="logprobs",
        max_tokens=_SHORT,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        logprobs=True,
        top_logprobs=5,
        compile_decode=True,
        materialize_logprobs_mlx_lm=True,
    ),
    "compile_decode_off": ScenarioSpec(
        id="compile_decode_off",
        max_tokens=_SHORT,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        compile_decode=False,
    ),
    "quantized_kv_8bit": ScenarioSpec(
        id="quantized_kv_8bit",
        max_tokens=_SHORT,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        compile_decode=True,
        quantized_kv_start=32,
        kv_bits=8,
        kv_group_size=64,
    ),
}

# Order for ``all`` — disciplined core matrix first, then compile/KV variants.
SCENARIO_ALL_ORDER: tuple[str, ...] = (
    "greedy_short",
    "greedy_long",
    "sampling_short",
    "sampling_long",
    "repetition_penalty",
    "logprobs",
    "compile_decode_off",
    "quantized_kv_8bit",
)


def cli_default_scenario(
    *,
    max_tokens: int,
    compile_decode: bool,
) -> ScenarioSpec:
    """Legacy single-scenario mode matching pre-scenario CLI behavior."""
    return ScenarioSpec(
        id="cli_default",
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        logprobs=False,
        top_logprobs=0,
        compile_decode=compile_decode,
    )


def parse_scenario_ids(spec: str) -> tuple[str, ...]:
    """Comma-separated ids, or ``all``."""
    raw = spec.strip()
    if not raw:
        return ()
    if raw.lower() == "all":
        return SCENARIO_ALL_ORDER
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    if not parts:
        return ()
    return parts


def resolve_scenarios(
    spec: str,
    *,
    max_tokens: int,
    compile_decode: bool,
) -> tuple[ScenarioSpec, ...]:
    """Return scenario tuple; empty spec → single CLI default."""
    ids = parse_scenario_ids(spec)
    if not ids:
        return (cli_default_scenario(max_tokens=max_tokens, compile_decode=compile_decode),)
    out: list[ScenarioSpec] = []
    for sid in ids:
        if sid == "cli_default":
            out.append(cli_default_scenario(max_tokens=max_tokens, compile_decode=compile_decode))
            continue
        if sid not in BUILTIN_SCENARIOS:
            known = ", ".join([*sorted(BUILTIN_SCENARIOS), "cli_default", "all"])
            raise ValueError(f"unknown scenario {sid!r}; known: {known}")
        out.append(BUILTIN_SCENARIOS[sid])
    return tuple(out)
