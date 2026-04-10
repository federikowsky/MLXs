"""Shared value types used across legacy generation-facing boundaries.

These types form the compatibility data contract between ``mlxs.generate``,
batch, server, and related legacy consumers. Phase 1 Layer 1 code in
``mlxs.runtime_core`` uses separate core-local contracts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class FinishReason(Enum):
    """Why generation stopped (FR9)."""

    STOP = auto()  # EOS token or stop sequence matched
    LENGTH = auto()  # max_tokens reached
    TOOL_CALLS = auto()  # tool call detected in output (FR7)
    CANCELLED = auto()  # request was cancelled by caller


class StreamPolicy(Enum):
    """MLX stream policy for generation (§6.8)."""

    SINGLE = "single"
    OVERLAP = "overlap"


class MemoryCeilingPolicy(Enum):
    """Action when memory ceiling is exceeded (§6.3.1)."""

    TRIM_CACHE = "trim_cache"
    REJECT_ONLY = "reject_only"
    SHUTDOWN = "shutdown"


class ModelMode(Enum):
    """Model execution mode (FR12, §7.4)."""

    TEXT = "text"
    MULTIMODAL = "multimodal"
    AUTO = "auto"


class PaddingSide(Enum):
    """Batch padding direction (§6.4, §8.2)."""

    LEFT = "left"
    RIGHT = "right"


class WeightFormat(str, Enum):
    """Weight/quantization format for model loading (FR1, §7.2).

    Extensible for future backends (AWQ, GPTQ, etc.). Only SAFETENSORS and
    PARO are implemented; others raise a clear error when selected.
    """

    AUTO = "auto"
    SAFETENSORS = "safetensors"
    PARO = "paro"
    AWQ = "awq"
    GPTQ = "gptq"


@dataclass(frozen=True, slots=True)
class TopLogprob:
    """A single top-logprob entry for a token position."""

    token_id: int
    token: str
    logprob: float


@dataclass(frozen=True, slots=True)
class TokenLogprobs:
    """Logprob information for one generated token (FR8)."""

    token_logprob: float
    top_logprobs: tuple[TopLogprob, ...] = ()


@dataclass(slots=True)
class TokenEvent:
    """One generated token in the legacy output stream.

    This remains the compatibility contract for legacy generate/server/batch
    surfaces. It is not the canonical Layer 1 output contract after Phase 1.
    """

    token_id: int
    text: str
    finish_reason: FinishReason | None = None
    logprobs: TokenLogprobs | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True, slots=True)
class GenerateOptions:
    """Options for a legacy single-request generation call (§6.1, §8.2).

    Passed to the compatibility ``generate()`` surface. It is not the
    canonical Layer 1 input contract after Phase 1.
    """

    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    min_p: float = 0.0  # 0 = disabled
    seed: int | None = None
    stop_sequences: tuple[str, ...] = ()
    extra_eos_token_ids: tuple[int, ...] = ()
    repetition_penalty: float = 1.0  # 1.0 = disabled
    logprobs: bool = False
    top_logprobs: int = 0  # 0 = disabled
    stream: bool = True


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Parsed tool call from model output (FR7, §6.6)."""

    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """Complete result of a non-streaming generation request."""

    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generation_tokens: int
    finish_reason: FinishReason
    tool_calls: tuple[ToolCallResult, ...] = ()
    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Per-request metrics collected during generation (§5.3)."""

    decode_tokens_per_second: float = 0.0
    prefill_tokens_per_second: float = 0.0
    time_to_first_token_seconds: float = 0.0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_cache_hit: bool = False
