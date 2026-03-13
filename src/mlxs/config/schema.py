"""Pydantic config schema — typed settings for all MLXs options (§8.2, AC3).

Every configurable option in specs.md §8.2 is represented here with its type,
default, and description. Config is frozen after construction (immutable).
Modules receive individual config sections, not the full AppConfig.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mlxs._types import (
    MemoryCeilingPolicy,
    ModelMode,
    PaddingSide,
    StreamPolicy,
    WeightFormat,
)


class _Frozen(BaseModel):
    """Base for all config sections — frozen after construction."""

    model_config = {"frozen": True, "extra": "forbid"}


class ModelConfig(_Frozen):
    """Model loading options (§8.2 Model)."""

    model_path: str = Field(
        default="",
        description="Local path or Hugging Face model id.",
    )
    adapter_path: str | None = Field(
        default=None,
        description="Optional adapter/LoRA path.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow trust_remote_code for tokenizer.",
    )
    model_type: str | None = Field(
        default=None,
        description="Model architecture type. Auto-detected from config.json if None.",
    )
    lazy_load: bool = Field(
        default=True,
        description="Load model on first use instead of at startup (FR11).",
    )
    preload: bool = Field(
        default=False,
        description="Load model at startup for predictable latency. Overrides lazy_load.",
    )
    model_mode: ModelMode = Field(
        default=ModelMode.AUTO,
        description="Execution mode: text-only, multimodal, or auto (FR12, §7.4).",
    )
    weight_format: WeightFormat = Field(
        default=WeightFormat.AUTO,
        description="Weight format: auto (detect), safetensors, paro (PARO/paroquant), "
        "awq, gptq (placeholders for future). NFR1, O4.",
    )
    model_hf_revision: str | None = Field(
        default=None,
        description="Hugging Face repo revision (branch, tag, or commit). Used when model_path is an HF id. Default uses Hub default (usually main).",
    )
    model_hf_token: str | None = Field(
        default=None,
        description="Hugging Face token for gated/private repos. Prefer HF_TOKEN env for security. Never logged.",
    )


class GenerateConfig(_Frozen):
    """Generation options (§8.2 Generation)."""

    max_tokens: int = Field(default=512, ge=0, description="Maximum tokens to generate.")
    temperature: float = Field(default=1.0, ge=0.0, description="Sampling temperature.")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold.")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling. 0 = disabled.")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Min-p sampling. 0 = disabled.")
    seed: int | None = Field(default=None, description="Random seed for reproducibility.")
    prefill_step_size: int = Field(
        default=2048,
        ge=1,
        description="Chunk size for prefill (§6.1).",
    )
    clear_cache_interval: int = Field(
        default=256,
        ge=0,
        description="Call mx.clear_cache() every N decode tokens. 0 = never (§6.8).",
    )
    stream: bool = Field(default=True, description="Stream tokens by default.")
    compile_decode: bool = Field(
        default=False,
        description="Use mx.compile on decode forward step (§6.8).",
    )
    warmup_after_load: bool = Field(
        default=False,
        description="Run dummy forwards after model load to warm the graph (§6.8).",
    )
    stream_policy: StreamPolicy = Field(
        default=StreamPolicy.SINGLE,
        description="MLX stream policy: single or overlap (§6.8).",
    )
    stop_sequences: tuple[str, ...] = Field(
        default=(),
        description="Additional stop sequences (FR9).",
    )
    extra_eos_token_ids: tuple[int, ...] = Field(
        default=(),
        description="Additional end-of-sequence token ids.",
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        description="Repetition penalty. 1.0 = disabled.",
    )
    logprobs: bool = Field(default=False, description="Return logprobs in stream.")
    top_logprobs: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Number of top logprobs per token. 0 = disabled.",
    )


class MemoryConfig(_Frozen):
    """Memory / MLX options (§8.2 Memory)."""

    wired_limit: int | None = Field(
        default=None,
        description="MLX wired memory limit in bytes. None = use device recommendation.",
    )


class CacheConfig(_Frozen):
    """KV cache options (§8.2 KV cache, §6.2)."""

    max_kv_size: int | None = Field(
        default=None,
        description="Max size for rotating cache (tokens). None = unlimited / non-rotating.",
    )
    kv_bits: int | None = Field(
        default=None,
        description="Quantization bits for KV cache. None = full precision.",
    )
    kv_group_size: int = Field(
        default=64,
        description="Group size for quantized KV cache.",
    )
    quantized_kv_start: int = Field(
        default=0,
        ge=0,
        description="Start quantizing KV after this many decode steps.",
    )
    kv_rotating_keep: int | None = Field(
        default=None,
        description="Tokens to keep in rotating cache. None = use max_kv_size default.",
    )


class PromptCacheConfig(_Frozen):
    """Prompt cache options (§8.2 Prompt cache, §6.3)."""

    enabled: bool = Field(default=True, description="Enable prompt prefix caching.")
    max_entries: int = Field(default=100, ge=0, description="Max cached prefixes (count).")
    max_bytes: int | None = Field(
        default=None,
        description="Max total bytes for cached KV state. None = no byte limit.",
    )
    trim_on_rss_gb: float | None = Field(
        default=None,
        description="Trim cache when process RSS exceeds this (GB). None = disabled.",
    )
    trim_on_pressure: int | None = Field(
        default=None,
        description="Trim on macOS memory pressure level (1=normal, 2=warn, 4=critical).",
    )
    trim_keep_entries: int = Field(
        default=1,
        ge=0,
        description="Minimum entries to keep during trim.",
    )
    trim_step: int = Field(
        default=1,
        ge=1,
        description="Entries to remove per trim iteration.",
    )
    target_rss_ratio: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Trim until RSS < max_rss * ratio.",
    )
    on_memory_ceiling: MemoryCeilingPolicy = Field(
        default=MemoryCeilingPolicy.TRIM_CACHE,
        description="Policy when memory ceiling exceeded (§6.3.1).",
    )
    persist_path: str | None = Field(
        default=None,
        description="Path to persist prompt cache to disk (optional).",
    )


class BatchConfig(_Frozen):
    """Batch options (§8.2 Batch, §6.4)."""

    prefill_batch_size: int = Field(
        default=1,
        ge=1,
        description="Max prompts in one prefill batch.",
    )
    completion_batch_size: int = Field(
        default=4,
        ge=1,
        description="Max concurrent sequences in decode batch.",
    )
    prefill_step_size: int = Field(
        default=2048,
        ge=1,
        description="Chunk size for batched prefill.",
    )
    padding_side: PaddingSide = Field(
        default=PaddingSide.LEFT,
        description="Padding side for batched inputs.",
    )
    time_budget_ms: int = Field(
        default=100,
        ge=0,
        description="Time budget per batch step (ms). 0 = size-based only.",
    )
    max_batch_size: int = Field(
        default=8,
        ge=1,
        description="Hard limit on batch size.",
    )


class SpeculativeConfig(_Frozen):
    """Speculative decoding options (§8.2 Speculative, §6.5)."""

    draft_model_path: str | None = Field(
        default=None,
        description="Path to draft model. None = speculative decoding disabled.",
    )
    num_draft_tokens: int = Field(
        default=5,
        ge=1,
        description="Number of draft tokens per verification step.",
    )


class ToolCallingConfig(_Frozen):
    """Tool calling options (§8.2 Tool calling, §6.6)."""

    tool_call_parser: str = Field(
        default="generic",
        description="Parser type: 'generic', 'qwen', or model-specific key.",
    )


class ServerConfig(_Frozen):
    """Server options (§8.2 Server, §6.7)."""

    host: str = Field(default="127.0.0.1", description="Bind address.")
    port: int = Field(default=8080, ge=1, le=65535, description="Listen port.")
    max_concurrent_requests: int = Field(
        default=16,
        ge=1,
        description="Max concurrent inference requests.",
    )
    max_queue_size: int = Field(
        default=64,
        ge=0,
        description="Max pending requests in queue. 0 = unbounded.",
    )
    request_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Request timeout in seconds.",
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of inference workers.",
    )


class ObservabilityConfig(_Frozen):
    """Observability options (§8.2 Observability, NFR2)."""

    log_level: str = Field(default="INFO", description="Logging level.")
    metrics_enabled: bool = Field(default=False, description="Enable metrics collection.")
    metrics_port: int = Field(default=9090, ge=1, le=65535, description="Metrics endpoint port.")


class AppConfig(_Frozen):
    """Root configuration — aggregates all config sections (§8).

    Constructed by ``config.resolve()`` after layered merging.
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    prompt_cache: PromptCacheConfig = Field(default_factory=PromptCacheConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    tool_calling: ToolCallingConfig = Field(default_factory=ToolCallingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    strict_validation: bool = Field(
        default=False,
        description="Fail on unknown config keys (§8.1).",
    )
