"""Shared value types used across module boundaries.

These types form the data contract between generate, batch, server, and other
consumers. They are pure data — no business logic, no MLX dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
    """One generated token in the output stream.

    This is the central data contract between generate and all consumers
    (server, batch, direct callers). Designed for minimal allocation in the
    decode loop — fields are set directly, not via constructor kwargs where
    avoidable.
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
    """Options for a single generation request (§6.1, §8.2).

    Passed to generate(); immutable after construction.
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
