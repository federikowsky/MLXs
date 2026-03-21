# Library specification — MLX inference & server

**Source of truth** for the custom MLX-based inference library and integrated server.  
Target: outperform mlx_lm on Apple Silicon (Mac) while remaining fully configurable and feature-complete.

**Version:** 0.3
**Status:** Draft
**Last updated:** 2026-03-18 (updated after specs-implementation alignment audit)

---

## 1. Objectives

| ID | Objective | Priority |
|----|-----------|----------|
| O1 | **Outperform mlx_lm** on decode tokens/s, prefill tokens/s, and batched throughput (requests/s) on the same hardware and model. | P0 |
| O2 | **Minimize Python overhead** in the hot path (decode loop, batch step): fewer allocations, minimal per-token work, explicit sync points. | P0 |
| O3 | **Full feature parity** with mlx_lm where relevant: KV cache (plain + quantized), prompt cache, batching, speculative decoding, tool calling, streaming. | P0 |
| O4 | **Super configurability**: every behavioral and performance knob exposed via config file, env vars, and CLI; no magic defaults without documentation. | P0 |
| O5 | **Clear architecture**: separate modules for load, generate, cache, batch, server; single responsibility; testable in isolation. | P1 |
| O6 | **Extensibility**: add new model architectures or features without tangling core inference. | P1 |
| O7 | **Code quality and Python efficiency**: follow the code quality standards (§2); minimize Python overhead; prefer data structures and operations that maximize efficiency and reduce overhead. | P0 |
| O8 | **Abstractions**: core components (load, generate, cache, batch, server) must depend on abstractions (interfaces/protocols); maximize isolation, modularity, and scalability. | P0 |

---

## 2. Code quality and Python efficiency

All Python code in the library must follow these principles and practices.

### 2.1 Code quality principles

- **Best practice and standards**: respect language and domain best practices, and project standards (conventions, style, established patterns).
- **Clean and clear**: explicit names, small functions, readable intent without redundant comments.
- **Efficient and performant**: avoid unnecessary work, appropriate data structures, suitable algorithms.
- **Optimized and compact**: no redundancy, no dead or duplicate code.
- **Scalable and modular**: components with well-defined responsibilities, explicit dependencies, extensibility without disrupting the rest.
- **SRP** (Single Responsibility Principle): one class/function = one reason to change.
- **DRY** (Don't Repeat Yourself): shared logic in one place only, reuse through functions/modules.
- **SOLID**: beyond SRP, respect Dependency Inversion, Open/Closed, Liskov Substitution, Interface Segregation where applicable.

### 2.2 Minimize Python overhead

- Prefer **data structures and operations** that maximize efficiency and reduce overhead (e.g. avoid per-token allocations in the decode loop, reuse buffers where possible).
- Prefer **C/Rust-optimized** libraries and built-in or stdlib implementations over reimplementing equivalent logic in pure Python when a faster alternative exists.

### 2.3 Use of Python built-ins and stdlib

- **Built-in functions**: use `map`, `any`, `all`, `filter`, `sum`, `min`, `max`, `sorted`, `enumerate`, `zip`, `reversed`, etc. instead of manual loops where they are clearer and faster.
- **C-level stdlib modules**: prefer implementations from `itertools`, `functools`, `collections`, `operator`, `math`, `statistics`, `heapq`, `bisect` over ad-hoc Python loops or reinventions.
- **Comprehensions and generators**: use list/dict/set comprehensions and generator expressions for compact, efficient iteration.
- **Native methods**: use native string and sequence methods (e.g. `str.join`, `list.extend`, `list.sort`) instead of reimplementing similar logic in pure Python.

### 2.4 Abstractions

- The codebase must work **through abstractions** (interfaces, protocols, abstract base classes, dependency injection where appropriate) so that:
  - **Isolation**: components depend on contracts, not concrete implementations; swapping implementations does not require changes in callers.
  - **Modularity**: each module has a clear boundary and a small, stable public surface.
  - **Scalability**: new model types, cache backends, or server transports can be added by implementing the same abstraction without modifying core inference or orchestration logic.
- Prefer explicit interfaces for load, generate, cache, and server entry points so that testing and alternative implementations (e.g. mocks, different backends) are straightforward.

---

## 3. Non-goals / out of scope

- Training, fine-tuning, LoRA, adapters (out of scope for this library; use mlx_lm or other tools).
- Quantization of **weights** (convert to 4-bit/8-bit): assume pre-quantized models (e.g. from mlx_lm or external convert); focus on **inference** and **KV cache** quantization. **Exception**: conversion of model weights between supported inference formats (layout normalisation, weight-format interoperability — not quantization) is in scope (see FR13).
- Non-Apple Silicon: target is **macOS + MLX**; other backends (CUDA, CPU-only) are out of scope unless explicitly added later.
- Graphical or browser-based chat UI: the server is API-first (OpenAPI-compatible, e.g. OpenAI-like); no graphical chat interface. An interactive CLI session for developer use and testing is in scope and distinct from a user-facing GUI.

---

## 4. Requirements

### 4.1 Functional requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| FR1 | Load models from local path or Hugging Face (by id); support config + safetensors. | Same as mlx_lm; required for compatibility. |
| FR2 | Tokenize input (string or token ids); support chat templates and optional system prompt. | Needed for chat and tool use. |
| FR3 | Generate text via: **single-request** (stream or non-stream) and **batched** (prefill + decode batch). | Core use cases. |
| FR4 | **KV cache**: full precision and **quantized** (configurable bits, group size, start step); optional **rotating** cache for long context. | Memory and long-context. |
| FR5 | **Prompt cache**: save/load KV cache by prefix; LRU eviction by count and/or bytes; optional trim on memory pressure / RSS. | Latency and throughput for multi-turn. |
| FR6 | **Speculative decoding**: draft model + verify; configurable number of draft steps. | Throughput. |
| FR7 | **Tool calling**: parse tool calls from output; configurable parsers per model family. | Feature parity. |
| FR8 | **Streaming**: token-by-token or chunk-by-chunk over HTTP (e.g. SSE); optional logprobs/top_logprobs. | UX and API compatibility. |
| FR9 | **Stop conditions**: EOS, custom stop sequences, max_tokens. | Standard behavior. |
| FR10 | **Sampling**: temperature, top_p, top_k, min_p, seed; optional logits processors (e.g. repetition penalty). | Quality and control. |
| FR11 | **Lazy model loading**: load model weights and build the graph only when first needed (e.g. first request or explicit load); configurable preload vs on-demand. | Startup time and memory: avoid loading unused models. |
| FR12 | **Model mode selection**: when the architecture supports it, allow running in **text-only** mode (vision/video/audio encoders disabled) to save memory and compute; optional **multimodal** mode when needed. | Efficiency for text-only workloads; flexibility for multimodal when required. |
| FR13 | **Model format conversion**: inspect source model weights, normalise layout, plan and execute conversion to an inference-compatible format, and verify the converted output against the source. Produces a verifiable manifest. Not for weight quantization (out of scope per §3). | Format interoperability; enables use of models in alternative weight formats without manual conversion steps. |

### 4.2 Non-functional requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| NFR1 | **Configurability**: all inference and server parameters documentable and overridable (default in code → config file → env → CLI). | No hidden magic. |
| NFR2 | **Observability**: structured logs; optional metrics (counters, histograms) for tokens/s, latency, cache hit, batch size. | Debug and tune. |
| NFR3 | **Stability**: bounded memory (wired limit, cache limits); no unbounded growth under sustained load. | Production. |
| NFR4 | **API contract**: server exposes a stable, versioned API (e.g. OpenAI-compatible or explicit version in path). | Integration. |

---

## 5. Performance targets and metrics

### 5.1 Baseline

- **Baseline** = mlx_lm (same version as used in project) on the same Mac, same model, same test prompts.
- All metrics must be **reproducible**. The benchmark must document: **mlx_lm version** (e.g. PyPI version or git tag), **model** (path or HF id), **test prompts** (corpus or length distribution and count), **prompt lengths** and **number of tokens to generate** for decode/prefill/TTFT, **batched scenario** (e.g. N=4 concurrent clients, request shape, duration), **macOS version**, **MLX version**, **number of runs** and **aggregation** (e.g. median over N runs). See AC8 for the benchmark script.
- **Internal regression baseline** for testing fallback behavior (§6.8): `stream_policy=single`, `compile_decode=false`, `warmup_after_load=false`.

### 5.2 Target: outperform mlx_lm

| Metric | Definition | Target |
|--------|-------------|--------|
| **Decode tokens/s** | Tokens generated per second in the decode phase (single request, no batching). | **≥ 1.10 × mlx_lm** (i.e. ≥ 10% faster). |
| **Prefill tokens/s** | Tokens processed per second during prefill (single request). | **≥ 1.05 × mlx_lm** (≥ 5% faster). |
| **Batched throughput** | Completed requests per second under concurrent load (e.g. N clients, same model). | **≥ 1.15 × mlx_lm** (≥ 15% higher). |
| **Time to first token (TTFT)** | Latency from request start to first token delivered. | **≤ mlx_lm** (no regression). |
| **Memory (RSS)** | Peak process RSS during a fixed benchmark (e.g. 100 tokens × 10 requests). | **≤ 1.05 × mlx_lm** (no significant increase). |

### 5.3 Metrics to expose and collect

- `decode_tokens_per_second` (per request and aggregate).
- `prefill_tokens_per_second` (per request).
- `requests_per_second` (batched).
- `time_to_first_token_seconds`.
- `peak_memory_gb` (process).
- `prompt_cache_hit_count`, `prompt_cache_miss_count`, `prompt_cache_eviction_count`.
- `batch_size_prefill`, `batch_size_decode` (histogram or percentiles).

These must be **measurable** in CI or a dedicated benchmark script (see §10, AC8).

---

## 6. Features (detailed)

### 6.1 Inference core

- **Model load**: path or HF id; adapter path optional; trust_remote_code for tokenizer.
- **Lazy loading**: model weights and graph are loaded on first use (first request or explicit `load()`); option to **preload** at startup for predictable latency. Configurable via `lazy_load` and `preload`.
- **Generation**:
  - Prefill in configurable **chunk size** (e.g. 512, 1024, 2048, 4096).
  - Decode loop: one forward per token (or per batch); **single stream** for generation (e.g. `mx.stream`).
  - Optional **async_eval** for overlap (note: `mx.async_eval` is experimental in MLX); **clear_cache** interval configurable (e.g. every N tokens).
- **Sampling**: temperature, top_p, top_k, min_p, seed; **logits processors** (e.g. repetition penalty, custom).
- **Stop**: EOS token ids; extra EOS tokens; stop sequences (list of strings); max_tokens.

### 6.2 KV cache

- **Types**: full-precision; **quantized** (bits, group_size, quantized_kv_start); **rotating** (max_size, keep); **chunked** (fixed-size KV chunks for architectures using chunked or sliding-window attention; configurable chunk size; enables long-context models that require bounded attention scope per chunk).
- **Note**: Combining **rotating** and **quantized** KV cache is not yet implemented in mlx_lm (NYI); the core supports both separately.
- **Batch variants**: same semantics for batched decode (left-padding or right-padding; configurable).
- **Trim**: where supported (e.g. rotating), trim by tokens or by bytes; document which caches support trim.

### 6.3 Prompt cache

- **Storage**: in-memory LRU; key = (model_id, prefix_token_ids); value = KV cache state.
- **Limits**: max entries (count); max total bytes (optional).
- **Eviction**: LRU when limit exceeded; optional **trim on RSS** or **trim on memory pressure** (macOS), with configurable thresholds.
- **Save/load**: optional persistence to disk (e.g. safetensors) for reuse across restarts (optional feature).

### 6.3.1 Memory monitoring and configurable behavior

- **Event-driven only**: Memory monitoring and reactions must be **event-driven**; no polling or dedicated loops that continuously monitor. Checks and trim run only in response to **events** (e.g. after each cache insert, after request completion, or on macOS memory-pressure dispatch events when using a system listener). This avoids background threads or timers that poll RSS or pressure.
- **Memory must be monitored** at those event points. Metrics: **process RSS** (e.g. via `resource.getrusage(RUSAGE_SELF).ru_maxrss`); optionally **system memory pressure** on macOS (e.g. 1=normal, 2=warning, 4=critical, via sysctl or **event-driven** dispatch-based listener as in mlx_lm server).
- **Configurable behavior when over threshold**: instead of shutting down or failing hard, the implementation must allow **policy-driven actions** via config flags and variables. Supported behavior (at least):
  - **Trim cache**: when RSS or memory pressure exceeds configured ceilings, **progressively trim the prompt cache** (and optionally call `mx.clear_cache()`) until below target (with headroom). Use configurable **trim_keep** (min entries to keep), **trim_step** (entries to remove per iteration), **target_rss_ratio** (trim until RSS is below max_rss × ratio, to keep headroom and avoid broken pipe). This mirrors the pattern in mlx_lm server (`_trim_cache_if_rss_over_ceiling`, `trim_to`).
  - **Reject only**: when over ceiling, reject new requests (e.g. 503) without trimming.
  - **Shutdown** (optional): only if explicitly configured; otherwise prefer trim or reject.
- **Last step — fail gracefully**: if even after **emptying the cache** (trim to zero / full clear) memory remains over ceiling, the implementation must **fail gracefully** (e.g. reject new requests with a defined error, return 503, log and optionally continue in degraded mode) instead of crashing or blocking. No silent undefined behavior.
- All thresholds and behaviors are **configurable** (see §8.2): max RSS (GB), max memory pressure (level), trim_keep, trim_step, target_rss_ratio, and policy (e.g. `on_memory_ceiling`: trim_cache | reject_only | shutdown). Default behavior should be **trim cache** to maximize availability without process exit.

### 6.4 Batching

- **Prefill batch**: multiple prompts in one forward; padding (left/right) and attention mask; configurable **prefill_batch_size** and **prefill_step_size**.
- **Decode batch**: multiple sequences decoded together; **completion_batch_size** (max concurrent in batch); scheduler: time-budget or size-based.
- **Continuous batching**: when a sequence finishes, remove from batch and add a new request if available (goal: high GPU utilization).

### 6.5 Speculative decoding

- **Draft model**: smaller/faster model; same tokenizer as target.
- **Steps**: configurable number of draft tokens per verification step.
- **Cache**: draft cache and target cache; rewind on reject.

### 6.6 Tool calling

- **Parse** model output for tool calls (JSON or model-specific format).
- **Configurable parsers** per model type (e.g. Qwen, Mistral, OpenAI-style).
- **Chat template**: optional tool message formatting.

### 6.7 Server

- **Transport**: HTTP; streaming via SSE or chunked body.
- **Endpoints**: at least chat/completions (stream and non-stream); health; optional metrics.
- **Concurrency**: multi-thread or async; single process; one or more “workers” that pull from a queue and run inference (configurable).
- **Backpressure**: reject or queue when at capacity; configurable max queue size and timeouts.

### 6.8 Core performance optimizations (model-agnostic)

All optimizations in this section apply to the **core** inference path and are **model-agnostic** (any autoregressive LLM). Model-specific optimizations (e.g. custom kernels for one architecture) belong in dedicated model modules and are out of scope for the core.

Each optimization is either **proven beneficial** or **fallback-safe**: when disabled or unsupported, behavior must equal the safe baseline with **no performance regression**.

| Optimization | Description | Fallback / default |
|--------------|-------------|---------------------|
| **Compiled decode step** | Optional `mx.compile` on the decode forward. Decode has fixed shape `(B, 1)` so the graph can be compiled once and reused. Config: `compile_decode` (bool). | When `false` or unsupported: use plain forward (same as today). |
| **Warmup run** | After model load (or on first use), optional 1–2 dummy forwards with typical shapes to capture/compile the graph. There is no MLX API for “graph capture”; warmup = running the forward. Config: `warmup_after_load` (bool). | When `false`: first real request may pay cold start; no other change. |
| **Clear cache interval** | How often to call `mx.clear_cache()` in the decode loop (e.g. every N tokens). Config: `clear_cache_interval` (int; 0 = never). If unsupported, behave as `clear_cache_interval=0`. | Default e.g. 256; tunable per deployment. |
| **Wired limit** | MLX wired memory limit; can be set from `device_info` or overridden. Config: `wired_limit` (optional int, bytes). Effective on macOS 15+ with Metal; default from `mx.device_info()["max_recommended_working_set_size"]`. | Default: use MLX device recommendation (current behavior). |
| **Buffer reuse in decode** | In the decode loop, reuse buffers for logits/sampled token when shape is fixed `(B, 1, V)`; avoid per-token allocations. | When batch size or shape changes: allocate as needed (no regression). |
| **Stream policy** | Default: **single** `generation_stream` for all generation work (prefill + decode). Optional: **overlap** — use multiple MLX streams (e.g. one for prefill, one for decode) and `mx.synchronize()` where needed so that prefill of the next batch can overlap with decode of the current one where the runtime benefits. Config: `stream_policy` (`"single"` \| `"overlap"`). **Regression test:** when overlap is enabled, decode tokens/s, TTFT, and batched throughput must not be worse than single-stream (e.g. median over ≥ N runs, same hardware and load). | When `single` or when overlap is disabled: use one stream only; behavior identical to baseline. If overlap is enabled but brings no benefit on a given machine, metrics must not be worse than single-stream. |

Implementations must not regress when an optimization is off or when the fallback path is used. Optimizations are exposed via config so they can be tuned or disabled per environment. **Internal regression baseline** for fallback tests: `stream_policy=single`, `compile_decode=false`, `warmup_after_load=false` (see AC12).

### 6.9 Failure handling and edge cases

- **Memory (OOM)**: Memory is **monitored** in an **event-driven** way (§6.3.1); when ceilings are exceeded, **behavior is configurable** via flags: e.g. trim prompt cache (with trim_keep, trim_step, target_rss_ratio). If even after **emptying the cache** (full trim) memory stays over ceiling, the implementation must **fail gracefully** (e.g. reject new requests, return a defined error or 503). When wired limit or system memory is exceeded and no headroom can be recovered by trimming, fail gracefully as above. Document behavior when RSS or wired limit is exceeded. Batch size and context length limits should be configurable to reduce OOM risk.
- **Model / tokenizer load failure**: Document response (e.g. error code, message) when model or tokenizer fails to load; optional retry or recovery behavior.
- **Request timeout**: Semantics of `request_timeout` (client wait vs inference abort) and HTTP response (e.g. 503, 429) when rejecting or when capacity is exceeded should be defined.
- **Edge cases**: Define or document behavior for: empty prompt; max_tokens=0; prompt exceeding context window; batch drain when model or config changes; concurrent model switch. These may be implementation notes or explicit requirements in a later revision.

---

## 7. Models

### 7.1 Supported architectures (target)

- **Phase 1**: Llama (Llama 3.x, 4.x), Qwen2 / Qwen3 (dense).
- **Phase 2**: Gemma, Mistral, Phi (as needed).
- **Extended architectures**: beyond Phase 1/2, additional architectures are supported as needed via the registry — including MoE variants (e.g. Mixtral, Qwen-MoE, DeepSeek), SSM-based (e.g. Mamba, RWKV), hybrid attention–SSM, and multimodal architectures. The registry is the authoritative list of supported `model_type` values at runtime.
- **Mechanism**: one implementation unit per architecture; **registry** (model_type → class pair). Adding a new architecture **does not require changes** to generate, cache, batch, or server — only a new architecture implementation and a registry entry.

### 7.2 Load path

- **Config**: `config.json` (or similar) with `model_type`, `hidden_size`, etc.
- **Weights**: safetensors (single or sharded; primary format); PARO/paroquant (alternative backend for paroquant-quantized weight layouts); auto-detection of format when `weight_format=auto`; support for mlx_lm-converted weight layouts.
- **Weight format selection**: configurable per deployment (`weight_format` option; `auto` | `safetensors` | `paro`; default `auto`). Auto-detection probes format from directory contents. Unrecognised or unsupported formats must fail with a clear error at load time.
- **Tokenizer**: from Hugging Face tokenizers; optional chat template from config or repo.

### 7.3 Compatibility

- **Weights**: prefer compatibility with mlx_lm-converted models (same layout) to avoid duplicate conversion pipelines.
- **API**: model must expose `__call__(input_ids, cache=..., mask=...)` and optional `make_cache()` for custom cache types.

### 7.4 Model mode (text-only vs multimodal)

- When the **architecture allows it** (e.g. VLM with separate vision encoder and LLM), support **selectable mode**:
  - **Text-only**: only the text/LLM stack is loaded and used; vision/video/audio encoders are not loaded or are bypassed. Reduces memory and avoids encoder compute for text-only requests.
  - **Multimodal**: full model with vision/video/audio when the request contains images, video, or other modalities.
- Mode is **configurable** (`model_mode: "text" | "multimodal" | "auto"`). In **auto** mode, the implementation detects whether the architecture provides a vision configuration; if present, mode resolves to multimodal; otherwise to text-only. Mode can be overridden per deployment; per-request override where the API allows.
- If the architecture does not support mode separation (e.g. dense text-only model), the option is a no-op or ignored.
- **Visual preprocessing** (multimodal mode): images are resized and tiled to fit within configurable pixel bounds before encoding. Relevant config options: `image_max_pixels` (upper bound on total pixels per image after resize); `image_min_pixels` (minimum pixels per tile or image; prevents over-downscaling); `max_images_per_request` (hard cap on simultaneous images per request; requests exceeding the cap are rejected before inference begins).
- **Video support**: architectures that support video input may accept one video per request; requests containing both images and video in the same input are rejected. Audio input is not supported in the current production scope.

---

## 8. Configuration

### 8.1 Principles

- **Layered override**: default in code → config file → environment variables → CLI. Each layer overrides the previous.
- **Documentation**: every option has a type, default, and short description in a single config schema (e.g. JSON Schema or Pydantic model).
- **Validation**: startup fails fast on unknown or invalid options (optional strict mode).

### 8.2 Categories of options

- **Model**: path / HF id; **model_hf_revision** (optional git revision for HF resolution); **model_hf_token** (optional HF access token for gated models); adapter_path; trust_remote_code; model_type (if not auto-detected); **weight_format** (auto | safetensors | paro; default auto); **lazy_load** (load on first use) vs **preload** (load at startup); **model_mode** (text | multimodal | auto); **image_max_pixels** (int; max pixels per image after resize); **image_min_pixels** (int; min pixels per tile or image); **max_images_per_request** (int; hard cap on simultaneous images per request).
- **Generation**: max_tokens; temperature; top_p; top_k; min_p; seed; **prefill_step_size** (int; default e.g. 2048); **clear_cache_interval** (int, tokens; default e.g. 256; 0 = never); stream; **compile_decode** (bool; use mx.compile on decode step); **warmup_after_load** (bool; dummy forwards after load); **stream_policy** (single | overlap; default single); **stop_sequences** (list of strings); **extra_eos_token_ids** (list of int, optional); **repetition_penalty** (float, optional) and/or logits processor options; **logprobs** / **top_logprobs** (optional, for streaming).
- **Memory / MLX**: **wired_limit** (optional bytes; default from device).
- **KV cache**: max_kv_size (rotating); kv_bits; kv_group_size; quantized_kv_start; **kv_rotating_keep** (int, optional; tokens to keep in rotating cache).
- **Prompt cache**: enabled; max_entries; max_bytes; trim_on_rss_gb; trim_on_pressure (macOS); trim_keep_entries; **trim_step** (entries to remove per trim iteration); **target_rss_ratio** (trim until RSS &lt; max_rss × ratio, e.g. 0.9); **on_memory_ceiling** (trim_cache | reject_only | shutdown; default trim_cache); **prompt_cache_persist_path** (optional; persist to disk).
- **Batch**: prefill_batch_size; completion_batch_size; prefill_step_size; **padding_side** (left | right); scheduler (time_budget_ms, max_batch_size).
- **Speculative**: draft_model_path; num_draft_tokens.
- **Tool calling**: **tool_call_parser** (enum or model-type key; configurable parser per model family).
- **Server**: host; port; max_concurrent_requests; max_queue_size; request_timeout; workers.
- **Observability**: log_level; metrics_enabled; metrics_port.
- **Validation**: **strict_validation** (bool; default false; if true, startup fails on unknown or invalid options).

### 8.3 Formats

- **Config file**: YAML or JSON; path via `--config` or env `CONFIG_PATH`.
- **Env**: prefix `MLX_INFER_`; e.g. `MLX_INFER_TEMPERATURE=0.7`. For nested config sections, use double-underscore (`__`) to separate section from key: e.g. `MLX_INFER_MODEL__WEIGHT_FORMAT=paro`, `MLX_INFER_SERVER__PORT=8080`.
- **CLI**: long and short flags; documented in `--help`.

---

## 9. Architecture (high-level modules)

Single responsibility; clear boundaries; testable in isolation.

| Module | Responsibility | Depends on |
|--------|----------------|------------|
| **config** | Load and merge config (file, env, CLI); validate; expose typed settings. | — |
| **protocols** | Inter-module contracts (Protocol classes / abstract interfaces); no implementations. | — |
| **shared types / errors** | Shared value types (token events, finish reasons, enums, logprobs); error hierarchy (load errors, memory errors, etc.). | — |
| **load** | Load model config + weights + tokenizer; model registry (model_type → class); format detection and dispatch (safetensors, PARO, …); optional HF resolution. | MLX, HF hub, local fs |
| **layers** | Reusable model building blocks: attention (SDPA), RoPE, activations, SSM, MoE, MLA; shared across architectures. | MLX |
| **models/** | Per-architecture model classes (e.g. Llama, Qwen, and extended architectures including MoE, SSM, multimodal); `__call__`, `make_cache`; no server or HTTP. | MLX, **cache**, **layers** |
| **cache** | KV cache types (full, quantized, rotating, chunked, batch variants); create, update, trim, serialize. | MLX |
| **generate** | Prefill + decode loop; sampling; logits processors; single-request only (no HTTP). | **cache**, **protocols** |
| **batch** | Batch scheduler; continuous batching; prefill batch + decode batch; add/remove sequences; call generate-style step. | **generate**, **cache** |
| **prompt_cache** | LRU of (prefix, KV cache); prefix trie for efficient matching; eviction by count/bytes; event-driven trim policies (RSS, pressure). | **cache** |
| **speculative** | Draft model + verify; rejection sampling; cache rewind on rejection; isolated from core generate. | **generate**, **cache** |
| **tool_calling** | Output parsers per model family; tool call detection and formatting; fully isolated from inference core. | — |
| **observability** | Structured logging; optional metrics (counters, histograms); no inference dependencies. | — |
| **model conversion** | Inspect, normalise, plan, execute, and verify conversion of model weights between supported inference formats; produces verifiable manifest. | **load** |
| **server** | HTTP API; SSE streaming; request queue and backpressure; dispatch to single (generate) or batched (batch) path; **composition root** — wires all concrete implementations. | **generate**, **batch**, **prompt_cache**, **load**, **config**, **observability** |
| **CLI session** | Interactive developer session; direct model load and generation without HTTP; for development and testing. | **generate**, **load**, **config** |

No circular dependencies. **Server** and **CLI** are the only composition roots that wire concrete implementations together; all other modules depend on protocols and abstractions, not on each other's concrete types.

**Interfaces:** Load, generate, cache, and server must expose stable contracts (e.g. load: path/model_id → model + tokenizer; generate: model, tokenizer, prompt, options → stream of tokens; cache: create/update/trim; server: request → response/stream). These contracts allow testing with mocks and swapping implementations without changing callers. A separate schema or appendix may detail method signatures and error handling.

---

## 10. Acceptance criteria

- [ ] **AC1** On at least one reference model (e.g. Llama 3.2 3B 4-bit) and one Mac, decode tokens/s ≥ 1.10 × mlx_lm in a documented benchmark.
- [ ] **AC2** On the same setup, batched throughput (e.g. 4 concurrent clients) ≥ 1.15 × mlx_lm.
- [ ] **AC3** All options in §8 (Configuration) are configurable via config file, env, and CLI and documented.
- [ ] **AC4** KV cache: full, quantized, and rotating implemented and tested.
- [ ] **AC5** Prompt cache: LRU with byte and count limits; at least one trim policy (RSS or pressure) implemented.
- [ ] **AC6** Server serves streaming and non-streaming chat/completions; TTFT and tokens/s measurable from client.
- [ ] **AC7** Code layout: separate packages/modules for load, generate, cache, batch, server; no single file > ~1500 lines for core logic. Modules are **testable in isolation** (interfaces allow mocks and alternative implementations).
- [ ] **AC8** Benchmark script (or CI job) runs and records metrics above; baseline mlx_lm run documented for comparison.
- [ ] **AC9** Code adheres to §2 (Code quality and Python efficiency): principles, minimal overhead, use of built-ins/stdlib, abstractions (§2.4).
- [ ] **AC10** Lazy model loading: model loads on first use when `lazy_load` is enabled; **preload** option loads at startup; both paths tested.
- [ ] **AC11** Model mode: for architectures that support it, text-only mode avoids loading/using vision/video; multimodal mode available and configurable; documented which models support mode selection.
- [ ] **AC12** Core optimizations (§6.8): implemented with fallback-safe behavior; when each optimization is disabled, performance is no worse than single-stream, uncompiled, no-warmup baseline; all options are model-agnostic and configurable.
- [ ] **AC13** Prefill tokens/s ≥ 1.05 × mlx_lm and TTFT ≤ mlx_lm (no regression) and peak RSS ≤ 1.05 × mlx_lm on the same benchmark setup as AC1/AC2.
- [ ] **AC14** Speculative decoding (FR6): implemented and tested; draft model + verify path works with configurable num_draft_tokens.
- [ ] **AC15** Tool calling (FR7): configurable parsers per model type; parsing and tool message formatting tested.
- [ ] **AC16** Observability (NFR2): structured logs and optional metrics (as in §5.3) implemented; stability (NFR3): wired and cache limits prevent unbounded growth; API (NFR4): server exposes a stable, versioned API.
- [ ] **AC17** Extensibility: a new model architecture can be added via the registry without modifying generate, cache, or server.
- [ ] **AC18** Model format conversion (FR13): inspect → normalise → plan → execute → verify pipeline produces a verifiable manifest; conversion of at least one supported format pair is tested end-to-end.

---

## 11. Glossary

- **Decode**: autoregressive step; one token per forward (or per batch).
- **Chunked KV cache**: KV cache variant where attention is computed over fixed-size chunks; used by architectures with chunked or sliding-window attention; bounds attention scope per chunk to enable long-context generation without unbounded memory growth.
- **Compiled decode**: optional use of `mx.compile` on the decode forward step (fixed shape); can reduce per-step overhead when enabled.
- **Lazy loading**: load model weights and build the compute graph only when first needed (e.g. first inference request or explicit load), not at process startup.
- **Model mode**: when the architecture allows, selectable execution mode—e.g. **text-only** (LLM only, no vision/video encoders) vs **multimodal** (full model including encoders).
- **Prefill**: process prompt in chunks; populate KV cache.
- **Prompt cache**: cache of (prefix, KV state) for reuse across requests with same prefix.
- **Rotating KV cache**: fixed-size cache; old tokens dropped (with optional keep).
- **Speculative decoding**: draft model proposes tokens; target model verifies; accept or rewind.
- **Stream policy**: whether to use a single MLX stream for generation (`single`) or optional multiple streams for overlap (`overlap`); fallback is always single-stream.
- **Sync point**: point in the execution where MLX/CPU synchronization is guaranteed (e.g. before/after `mx.eval`, or at batch boundaries); used to minimize Python overhead and make timing predictable (see O2).
- **Trim cache (on memory ceiling)**: when process RSS or system memory pressure exceeds configured thresholds (checked **event-driven**, no polling), progressively remove prompt-cache entries (with trim_keep, trim_step, target_rss_ratio) instead of shutting down; if even full cache empty does not resolve, fail gracefully. Configurable via `on_memory_ceiling` (trim_cache | reject_only | shutdown).
- **TTFT**: time to first token.
- **Warmup run**: optional dummy forward(s) after load to capture/compile the graph so the first real request avoids cold start.
- **Wired memory**: memory locked by process (e.g. MLX wired limit on macOS).

---

## 12. References

- mlx_lm: `generate.py`, `server.py`, `models/cache.py`, `utils.py`. For memory monitoring and trim-on-ceiling behavior: `server.py` (`_process_rss_gb`, `_memory_pressure_level`, `_trim_cache_if_rss_over_ceiling`, `trim_to`, `--prompt-cache-max-rss`, `--prompt-cache-max-pressure`, `--prompt-cache-trim-keep`, `--prompt-cache-trim-step`, `--prompt-cache-target-rss-ratio`).
- MLX: `mx.stream`, `mx.new_stream`, `mx.eval`, `mx.async_eval` (experimental), `mx.synchronize`, `mx.clear_cache`, `mx.set_wired_limit`, `mx.compile`, `mx.device_info`, `mx.get_peak_memory`. Wired limit is effective on macOS 15+ with Metal.
- Project doc: `docs/kv-cache-and-prompt-cache.md` (prompt cache and KV options).

---

*This document is the single source of truth for scope, requirements, performance targets, and configuration. Implementation tickets and ADRs should reference section and requirement IDs.*
