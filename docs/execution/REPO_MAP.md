# Repo Map

## Benchmark surfaces

- Canonical benchmark entry:
  - `benchmarks/mlxs_vs_mlx_lm/__main__.py`
  - `benchmarks/mlxs_vs_mlx_lm/harness.py`
  - `benchmarks/mlxs_vs_mlx_lm/backends.py`
- Benchmark support:
  - `discovery.py`
  - `json_util.py`
  - `memory.py`
  - `stats.py`
  - `system_info.py`

## Layer 1 — performance core

- Runtime core:
  - `src/mlxs/runtime_core/decode.py`
  - `src/mlxs/runtime_core/prefill.py`
  - `src/mlxs/runtime_core/state.py`
  - `src/mlxs/runtime_core/selection.py`
  - `src/mlxs/runtime_core/greedy.py`
  - `src/mlxs/runtime_core/contracts.py`
  - `src/mlxs/runtime_core/policy.py`
- Cache substrate:
  - `src/mlxs/cache/kv.py`
  - `src/mlxs/cache/attention_mask.py`
  - `src/mlxs/cache/chunked.py`
  - `src/mlxs/cache/quantized.py`
  - `src/mlxs/cache/rotating.py`

## Layer 2 — generation semantics

- `src/mlxs/general_path/single_request.py`
- `src/mlxs/general_path/finish.py`
- `src/mlxs/general_path/stop.py`
- `src/mlxs/generate/decode.py`
- `src/mlxs/generate/prefill.py`
- `src/mlxs/generate/logits.py`
- `src/mlxs/generate/sampling.py`
- `src/mlxs/generate/compile.py`

## Layer 3 — advanced engines

- Batch:
  - `src/mlxs/batch/scheduler.py`
  - `src/mlxs/batch/streams.py`
- Prompt cache:
  - `src/mlxs/prompt_cache/lru.py`
  - `src/mlxs/prompt_cache/memory.py`
- Speculative:
  - `src/mlxs/speculative/draft.py`
  - `src/mlxs/speculative/verify.py`

## Layer 4 — product surfaces

- `src/mlxs/product_surfaces/bootstrap.py`
- `src/mlxs/product_surfaces/batched_serving.py`
- `src/mlxs/product_surfaces/chat.py`
- `src/mlxs/product_surfaces/http.py`
- `src/mlxs/product_surfaces/cli.py`
- `src/mlxs/product_surfaces/compat_openai.py`
- `src/mlxs/chat/cli.py`
- `src/mlxs/chat/tui/__init__.py`
- `src/mlxs/chat/tui/completion.py`

## Model/load surfaces

- Registry and load path:
  - `src/mlxs/load/registry.py`
  - `src/mlxs/load/loader.py`
  - `src/mlxs/load/resolve.py`
  - `src/mlxs/load/weights.py`
  - `src/mlxs/load/tokenizer.py`
- Current compatibility-sensitive models:
  - `src/mlxs/models/qwen.py`
  - `src/mlxs/models/llama.py`
  - `src/mlxs/models/multimodal_shared.py`

## Critical tests

- Benchmark surfaces:
  - `tests/unit/test_benchmarks/test_backends.py`
  - `tests/unit/test_benchmarks/test_harness.py`
- Runtime/general-path:
  - `tests/unit/test_general_path/test_single_request.py`
  - `tests/unit/test_general_path/test_boundaries.py`
  - `tests/unit/test_generate/test_compile.py`
- Batch/product surfaces:
  - `tests/unit/test_batch/test_scheduler.py`
  - `tests/unit/test_product_surfaces/test_batched_serving.py`
  - `tests/unit/test_product_surfaces/test_layer4_boundaries.py`
- Qwen compatibility:
  - `tests/unit/test_models/test_qwen2_forward.py`

## Accepted promoted files

- `benchmarks/mlxs_vs_mlx_lm/backends.py`
- `benchmarks/mlxs_vs_mlx_lm/harness.py`
- `src/mlxs/general_path/single_request.py`
- `src/mlxs/generate/compile.py`
- `src/mlxs/chat/cli.py`
- `src/mlxs/chat/tui/__init__.py`
- `src/mlxs/chat/tui/completion.py`
- `src/mlxs/models/qwen.py`
- `src/mlxs/product_surfaces/batched_serving.py`
