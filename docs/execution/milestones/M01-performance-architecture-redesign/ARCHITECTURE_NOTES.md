# M01.R1 Architecture Notes

Updated: 2026-04-19

## MLXs hot-path map

### Layer 1 single-request core

`src/mlxs/runtime_core/prefill.py`

1. `run_prefill()`
   - owns prompt ingestion
   - chunks prompt to `prefill_step_size`
   - evaluates cache states with `mx.eval(cache_states)`
   - clears cache after each chunk
   - returns last-step logits only

`src/mlxs/runtime_core/decode.py`

2. `decode_step()`
   - resolves step function
   - selects token
   - `mx.eval(token)`
   - `token.item()`
   - updates generation counter and finish signal
   - clears cache on interval
   - if not finished, runs next forward and returns next logits

3. prepared-step helpers
   - `prepare_decode_step()`: select token, optional `mx.async_eval(token)`
   - `prepare_next_logits()`: next forward, optional `mx.async_eval(next_logits)`
   - `schedule_next_decode_step()`: next forward + next token selection
   - `materialize_prepared_step()`: grouped `mx.eval(prepared.token, next_prepared.token)` or forced eval, then `item()`

4. retained neutral helpers
   - `_decode_step_with_resolved_step()`
   - `_prepare_next_logits_with_resolved_step()`
   - `_schedule_next_decode_step_with_resolved_step()`

### Canonical Layer 1 entrypoint

`src/mlxs/runtime_core/greedy.py`

- `run_greedy()`
  - `CoreState.create()`
  - `run_prefill()`
  - plain `while True: decode_step(...)`
  - no prepared-step/lookahead gate

### Benchmark Class A MLXs helper

`benchmarks/mlxs_vs_mlx_lm/backends.py::_run_mlxs_class_a_trial`

- creates `CoreState`, `CoreExecutionPolicy`, `CoreTerminationPolicy`
- optional compiled `step_fn`
- `run_prefill()`
- branches:
  - lookahead branch:
    - `prepare_decode_step()`
    - `_schedule_next_decode_step_with_resolved_step()`
    - `materialize_prepared_step()`
  - baseline branch:
    - `decode_step()`
- model-specific lookahead gate:
  - prompt `<=512`
  - all `qwen2`
  - `llama` with `hidden_size >= 3072`

### General path

`src/mlxs/general_path/single_request.py`

- owns several separate decode progressions:
  - prepared-step enriched path
  - sampled prepared-step path
  - processor prepared-step path
  - baseline decode path
- does:
  - text decode
  - stop-sequence text matching
  - finish mapping
  - logprob shaping
  - `TokenEvent` creation

### Legacy compatibility decode loop

`src/mlxs/generate/decode.py`

- owns another independent decode loop
- synchronous `mx.eval(y, logprobs)` each step
- per-token `TokenEvent`
- optional logits processors, logprobs, quantized-KV transitions

### Batch scheduler

`src/mlxs/batch/scheduler.py`

- prefill:
  - `_prefill_one()` and `_prefill_cohort()`
  - chunked prompt processing
  - first token sampled and immediately shaped into `TokenEvent`
- decode:
  - `_decode_active()`
  - single-request path:
    - model forward on `current_token`
    - per-seq processors
    - sample
    - `mx.eval(y)`
    - `y.item()`
    - text decode and `TokenEvent`
  - grouped path:
    - `_merge_group_cache()`
    - batched forward
    - per-seq processors and sampling
    - `mx.eval([*next_tokens,*merged_states])`
    - `_scatter_group_cache()`
    - per-seq `y.item()`, text decode, `TokenEvent`

## `mlx_lm` hot-path map

### Single-request generation

`mlx_lm/generate.py::generate_step`

1. prompt prefill
   - `_model_call()` in `with mx.stream(generation_stream)`
   - chunked prefill loop
   - `mx.eval([c.state for c in prompt_cache])`
   - `mx.clear_cache()` after prompt chunks

2. `_step()`
   - model forward under generation stream
   - processors applied inside `_step`
   - `quantize_cache_fn(prompt_cache)`
   - `logprobs = logits - logsumexp`
   - `sampled = sampler(logprobs)`

3. decode loop
   - initial `mx.async_eval(y, logprobs)`
   - each iteration:
     - compute `next_y, next_logprobs = _step(y)` before yielding current token
     - `mx.async_eval(next_y, next_logprobs)`
     - on first token only: `mx.eval(y)`
     - yield `y.item(), logprobs`
     - `mx.clear_cache()` every 256 iterations
     - advance `y, logprobs = next_y, next_logprobs`

### Response shaping

`mlx_lm/generate.py::stream_generate`

- tokenization / detokenizer
- `generate_step()` / speculative generator as the token source
- response shaping is above the core token generator

### Batch generation

`mlx_lm/generate.py::BatchGenerator`

- resident `active_batch` owns:
  - `y`
  - `logprobs`
  - `cache`
  - `tokens`
  - `uids`
  - `num_tokens`
  - `max_tokens`
  - `samplers`
  - `logits_processors`
- `_next()`:
  - may process prompts into a `Batch`
  - if completions already active, evaluates old `batch.y, batch.logprobs`
  - updates `batch.tokens`
  - computes next `batch.y, batch.logprobs = self._step(...)`
  - `mx.async_eval(batch.y, batch.logprobs, batch.tokens)`
  - materializes prior outputs with `y.tolist()`
  - filters or extends the resident batch in place

## MLX substrate constraints/opportunities

### Constraints from official docs

- lazy evaluation:
  - graph builds happen before `eval`
  - unused outputs still build graph and carry cost
- `eval`:
  - explicit synchronization/evaluation boundary
  - fixed overhead per evaluation
  - graph-size overhead also grows with graph size
- `async_eval`:
  - asynchronous evaluation trigger
  - experimental API
- streams:
  - operations and RNG accept a `stream`
  - stream choice is explicit and should stay an ownership concern
- compile:
  - caches compiled functions
  - recompiles on shape/type/input-count changes
  - cannot safely evaluate/print arrays inside compiled functions
  - outermost stable function is preferred
- device/memory:
  - `clear_cache`, `get_peak_memory`, `set_wired_limit`, `device_info` are real operational levers

### Opportunities implied by the substrate

- move expensive wrapper resolution out of per-token loops
- keep next-step work resident and asynchronously launched where safe
- keep output contracts minimal so unused richer outputs do not build graphs
- compile only once stable shapes/owners are established

## Structural delta summary

1. MLXs benchmark helper now has a safe promoted lookahead branch; the real core entrypoint does not.
2. MLXs still owns multiple decode loops across Layer 1, Layer 2, legacy generate, and batch.
3. `mlx_lm` keeps both single-request and batch progression resident inside one owner each.
4. MLXs batch path still rebuilds grouping/cache state and shapes events inside the scheduler loop.
5. MLX itself rewards explicit evaluation discipline and resident progression, and punishes extra graph/eval boundaries.
6. MLXs has no batch-aware cache substrate today; `mlx_lm` batch residency depends on cache objects that support `merge`, `filter`, `extend`, and `extract`, while MLXs `KVCache` only exposes per-request mutation and serialization.

## R20 Model Audit Note

See [MODEL_AUDIT.md](MODEL_AUDIT.md) for the focused `Llama` and `Qwen` correctness sanity checks used during the weakest-case AC1 decomposition.
