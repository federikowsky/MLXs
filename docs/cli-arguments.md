<!-- Generated from src/mlxs/config/cli.py _CLI_SPECS. Regenerate: python -c "from pathlib import Path; from mlxs.config.cli import build_cli_arguments_md; Path('docs/cli-arguments.md').write_text(build_cli_arguments_md())" -->

# CLI arguments — full reference

All configuration options can be passed via explicit flags. Flags are available for both `mlxs serve` and `mlxs chat` unless noted. Values are merged with config file and env (CLI overrides win).

`mlxs chat` runs the interactive docked shell by default. Use `mlxs chat QUERY` for one-shot mode.
In a TTY, the shell includes slash commands, completion, and `@file` attachments.

## Chat usage

| Command | Description |
|---------|-------------|
| `mlxs chat` | Start the interactive docked chat shell. |
| `mlxs chat QUERY` | Run one turn and exit. |

Interactive mode includes slash commands such as `/help`, `/model`, `/history`, `/export`, `/new`, `/clear`, `/retry`, `/system`, `/stats`, and `/quit`, plus `@file` attachments.

---

## Config file

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | path | - | Path to YAML config file. Overrides CONFIG_PATH env. |

Generic overrides (any key): `-o section.key=value` (e.g. `-o server.port=9090`). May be repeated.

---

## Model (model.*)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | "" | Local path or Hugging Face model id |
| `--adapter` | str | None | Optional adapter/LoRA path |
| `--trust-remote-code` | bool | False | Allow trust_remote_code |
| `--no-trust-remote-code` | - | - | Disable (default). |
| `--model-type` | str | None | Model architecture (auto-detected if unset) |
| `--lazy-load` | bool | True | Load model on first use |
| `--no-lazy-load` | - | - | Disable (default). |
| `--preload` | bool | False | Load model at startup |
| `--no-preload` | - | - | Disable (default). |
| `--model-mode` | choice | auto | Execution mode: text, multimodal, auto |
| `--weight-format` | choice | auto | Format: auto, safetensors, paro, awq, gptq |
| `--model-hf-revision` | str | None | Hugging Face repo revision |
| `--model-hf-token` | str | None | Hugging Face token for gated repos |

---

## Generation (generate.*)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-tokens` | int | 512 | Maximum tokens to generate |
| `--temperature` | float | 1.0 | Sampling temperature |
| `--top-p` | float | 1.0 | Nucleus sampling threshold |
| `--top-k` | int | 0 | Top-k sampling (0=disabled) |
| `--min-p` | float | 0.0 | Min-p sampling (0=disabled) |
| `--seed` | int | None | Random seed |
| `--prefill-step-size` | int | 2048 | Prefill chunk size |
| `--clear-cache-interval` | int | 256 | mx.clear_cache every N tokens (0=never) |
| `--stream` | bool | True | Stream tokens by default |
| `--no-stream` | - | - | Disable (default). |
| `--compile-decode` | bool | False | Use mx.compile on decode |
| `--no-compile-decode` | - | - | Disable (default). |
| `--warmup-after-load` | bool | False | Warm graph after model load |
| `--no-warmup-after-load` | - | - | Disable (default). |
| `--stream-policy` | choice | single | Stream policy: single, overlap |
| `--repetition-penalty` | float | 1.0 | Repetition penalty (1.0=disabled) |
| `--logprobs` | bool | False | Return logprobs in stream |
| `--no-logprobs` | - | - | Disable (default). |
| `--top-logprobs` | int | 0 | Top logprobs per token (0=disabled) |
| `--stop` | str (repeat) | - | Stop sequence. May be repeated. |
| `--extra-eos-token-ids` | str | - | Comma-separated token ids (e.g. 1,2,3). |

---

## Memory (memory.*)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--wired-limit` | int | None | MLX wired memory limit (bytes) |

---

## KV cache (cache.*)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-kv-size` | int | None | Max rotating cache size (tokens) |
| `--kv-bits` | int | None | KV cache quantization bits |
| `--kv-group-size` | int | 64 | Group size for quantized KV |
| `--quantized-kv-start` | int | 0 | Start quantizing after N steps |
| `--kv-rotating-keep` | int | None | Tokens to keep in rotating cache |

---

## Prompt cache (prompt_cache.*)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-cache-enabled` | bool | True | Enable prompt prefix caching |
| `--no-prompt-cache-enabled` | - | - | Disable (default). |
| `--prompt-cache-max-entries` | int | 100 | Max cached prefixes |
| `--prompt-cache-max-bytes` | int | None | Max bytes for cached KV |
| `--prompt-cache-trim-on-rss-gb` | float | None | Trim when RSS exceeds (GB) |
| `--prompt-cache-trim-on-pressure` | int | None | Trim on memory pressure (1,2,4) |
| `--prompt-cache-trim-keep-entries` | int | 1 | Min entries to keep on trim |
| `--prompt-cache-trim-step` | int | 1 | Entries to remove per trim |
| `--prompt-cache-target-rss-ratio` | float | 0.9 | Trim until RSS < max*ratio |
| `--prompt-cache-on-memory-ceiling` | choice | trim_cache | Policy: trim_cache, reject_only, shutdown |
| `--prompt-cache-persist` | str | None | Path to persist cache |

---

## Batch (batch.*) — mainly for serve

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--batch-prefill-batch-size` | int | 1 | Max prompts per prefill batch |
| `--batch-completion-batch-size` | int | 4 | Max concurrent decode sequences |
| `--batch-prefill-step-size` | int | 2048 | Batched prefill chunk size |
| `--batch-padding-side` | choice | left | Padding: left, right |
| `--batch-time-budget-ms` | int | 100 | Time budget per batch step (ms) |
| `--batch-max-batch-size` | int | 8 | Hard limit on batch size |

---

## Speculative (speculative.*)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--draft-model` | str | None | Path to draft model (disables if unset) |
| `--num-draft-tokens` | int | 5 | Draft tokens per verification step |

---

## Tool calling (tool_calling.*)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--tool-call-parser` | str | generic | Parser: generic, qwen, or model key |

---

## Server (server.*) — for `mlxs serve`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--host` | str | 127.0.0.1 | Bind address |
| `--port` | int | 8080 | Listen port |
| `--max-concurrent-requests` | int | 16 | Max concurrent requests |
| `--max-queue-size` | int | 64 | Max pending requests (0=unbounded) |
| `--request-timeout` | float | 300.0 | Request timeout (seconds) |
| `--workers` | int | 1 | Number of workers |

---

## Observability (observability.*)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--log-level` | str | INFO | Logging level |
| `--metrics-enabled` | bool | False | Enable metrics |
| `--no-metrics-enabled` | - | - | Disable (default). |
| `--metrics-port` | int | 9090 | Metrics endpoint port |

---

## Root

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--strict-validation` | bool | False | Fail on unknown config keys |
| `--no-strict-validation` | - | - | Disable (default). |

---
