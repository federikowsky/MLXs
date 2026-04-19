# M01.R1 Deep Architecture Research

Updated: 2026-04-19

## Phase 0 — Framing

### Exact objective

- Reconstruct the full hot-path architecture of MLXs, `mlx_lm`, and MLX.
- Identify the real structural taxes behind the remaining AC1/AC2 gaps.
- End with one ranked redesign thesis, one fallback thesis, and one first bounded implementation slice.
- No new performance implementation slice is allowed until this research phase is complete.

### Why this matters

- AC1 is still open on the broadened accepted set.
- AC2 is still open with an authoritative scheduler baseline behind `mlx_lm`.
- Multiple bounded seam families were already exhausted or rejected.
- Recent wins on Qwen and Llama 3B are benchmark-surface wins, not yet core-architectural wins.

### What is already known

- MLXs has a clean Layer 1/2/3/4 architectural intent.
- Benchmark helper lookahead wins can be real and safe on accepted-sensitive remote slices.
- The accepted benchmark helper now uses retained resolved-step helpers on the lookahead branch only.
- `runtime_core.run_greedy` still uses plain `decode_step`.
- `general_path` duplicates several prepared-step loops.
- `batch.scheduler` owns its own prefill/decode/materialization/event loop and rebuilds group state.
- `mlx_lm` keeps a resident generation loop and a resident batch object.

### What is already exhausted/frozen

- General-path short enriched `logprobs/top_logprobs` shaping overhead family.
- AC13 bounded prefill tuning and local prefill candidates.
- AC2 merge/scatter elimination, grouped greedy fast path, resident active batch progression, async next-step progression, broad batch-aware cache contract, lightweight event materialization.
- Rejected redesign shapes:
  - class-based resident progression container
  - data-only progression container
  - inline helper migration across both benchmark decode branches

## Phase 1 — Sources Used

### Repo surfaces

- `src/mlxs/runtime_core/{prefill.py,decode.py,greedy.py,state.py,policy.py,contracts.py}`
- `src/mlxs/general_path/single_request.py`
- `src/mlxs/batch/scheduler.py`
- `src/mlxs/generate/decode.py`
- `benchmarks/mlxs_vs_mlx_lm/backends.py`
- `tests/unit/test_runtime_core/*`
- `tests/unit/test_general_path/test_boundaries.py`
- `tests/unit/test_benchmarks/{test_backends.py,test_harness.py}`

### Accepted artifacts

- `/tmp/mlxs_class_a_accepted_remote_20260418_1.json`
- `/tmp/mlxs_class_a_qwen15b_promoted_remote_20260418_3.json`
- `/tmp/mlxs_class_a_llama32_3b_promoted_remote_20260418_1.json`
- `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`
- `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`
- `/tmp/mlxs_ac13_prefillstep_ranked_20260418_1.json`
- `/tmp/mlxs_m01_lookahead_branch_promoted_remote_20260419_1.json`
- `/tmp/mlxs_m01_lookahead_branch_promoted_compare_20260419_1.json`

### Upstream local sources

- `mlx_lm/generate.py` from installed `mlx_lm`
- installed `mlx.core` runtime/docstrings

### Official MLX sources/docs

- Lazy Evaluation: <https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html>
- Compilation: <https://ml-explore.github.io/mlx/build/html/usage/compile.html>
- Using Streams: <https://ml-explore.github.io/mlx/build/html/usage/using_streams.html>
- Transforms (`eval`, `async_eval`, `compile`): <https://ml-explore.github.io/mlx/build/html/python/transforms.html>

## Phase 2 — Architecture Maps

Detailed maps are in [ARCHITECTURE_NOTES.md](ARCHITECTURE_NOTES.md).

## Phase 3 — Divergence Map

| Dimension | MLXs today | `mlx_lm` today | Likely effect |
| --- | --- | --- | --- |
| Single-request core owner | `runtime_core.run_greedy` plain `decode_step`; benchmark helper carries promoted lookahead branch | `generate_step` owns resident async next-step loop directly | MLXs win is stranded above the real core |
| Lookahead progression | helper-only, gated by prompt/model; not in canonical core entrypoint | resident in core generation loop | structural gap between benchmark win and core reality |
| Output contract | Layer 1 minimal `CoreStepResult`; richer shaping above | `generate_step` yields token + logprobs; `stream_generate` shapes response above | broadly aligned on single-request minimality |
| State persistence | `CoreState` owns cache; current logits survive in loop variable; no shared resident progression across consumers | current token/logprobs and prompt-cache survive inside one resident loop; batch has resident `Batch` object | MLXs duplicates progression ownership across consumers |
| Host/device sync | Layer 1 uses `mx.eval(token)` / `item()`; helper uses lookahead eval pairing; batch syncs sampled tokens and merged cache states | single-request: `mx.async_eval(next_y,next_logprobs)` then `mx.eval(y)` only when needed; batch: `mx.async_eval(batch.y, batch.logprobs, batch.tokens)` then `y.tolist()` | `mlx_lm` is more aggressively resident and asynchronous |
| Batch progression | `BatchScheduler` rebuilds groups, merges caches, scatters caches, samples per-sequence, emits `TokenEvent` in scheduler loop | `BatchGenerator` keeps resident `Batch{y,logprobs,cache,tokens,...}` and filters/extends in place | AC2 gap is structural, not merge/scatter alone |
| Layer 2 enrichment | `general_path` duplicates prepared-step loops and does text/logprob shaping inline | `stream_generate`/`generate` sit above `generate_step`; generation step remains minimal | MLXs still pays for multi-owner progression logic |
| Processor/logprobs path | processors/logprobs handled in `general_path` and legacy `generate.decode_loop`; benchmark class isolates them away | processors and samplers integrated into `_step`; broader shaping above | single-request fast path advantage is mostly outside these paths |
| Compile seam | prompt-aware `<=512` gate; compile still secondary and shape-sensitive | upstream eager comparable path is primary | compile is not the current main lever |

## Phase 4 — Bottleneck Model

### Architectural taxes

1. `benchmark/core split`
   - real accepted wins still live in the benchmark helper, not in `runtime_core.run_greedy`
   - this is the clearest architectural tax today

2. `loop ownership duplication`
   - benchmark helper, `runtime_core.run_greedy`, `general_path`, `generate.decode_loop`, and `batch.scheduler` each advance decode differently
   - this makes every optimization local unless ownership is unified

3. `batch resident-state deficit`
   - `BatchScheduler` does not keep a resident batch object equivalent to `mlx_lm.Batch`
   - it rebuilds/merges/scatters state and performs product shaping inside the scheduler loop
   - MLXs also lacks the missing substrate beneath that owner: a batch-aware cache contract with row-level `filter` / `extend` / `extract` semantics

4. `early shaping above/beside the core`
   - Layer 2 and Layer 3 still shape text/logprobs/events in or adjacent to hot loops
   - not the primary single-request tax, but a real structural tax for general path and AC2

5. `compile shape sensitivity`
   - compile remains useful but secondary; prompt-length-aware gating already proved shape sensitivity matters
   - compile-first redesign would solve the wrong problem

### Incidental taxes

- specific helper wrapper overheads
- local object/container dispatch in hot loops
- branch-local wrapper indirection

These matter, but only after the structural owner of the loop is correct.

## Phase 5 — Ranked Redesign Thesis

### Thesis 1 — Core-first resident progression unification

Build a real Layer 1 progression kernel that becomes the single owner of single-request decode advancement.

It should:

- live in `runtime_core`
- keep the loop inline or near-inline, not behind a hot-loop container/method object
- carry only minimal core data
- support the promoted lookahead branch as a real core path
- become the substrate for:
  - `run_greedy`
  - benchmark helper
  - later, selected `general_path` prepared-step consumers
  - later, a batch progression redesign

Why ranked first:

- strongest leverage on AC1 today
- closes the benchmark/core split
- improves architecture clarity even if speedup is modest
- creates a real foundation for AC2 instead of another helper-only win

### Thesis 2 — Fallback: batch-first resident progression redesign

If `run_greedy` migration proves neutral but not leverage-rich, redesign Layer 3 around a resident batch object modeled more closely on `mlx_lm.BatchGenerator`.

Why fallback only:

- AC2 needs it eventually
- but AC1/core still lacks a real core-owned progression win
- doing batch-first now risks widening the wrong ownership before Layer 1 is stabilized

## What should not be attempted

- more container-shaped progression objects in the hot loop
- another broad helper-only rewrite across both decode branches
- `general_path` first
- `batch.scheduler` first
- compile-first redesign
- new model-specific benchmark-helper gates as a substitute for architectural change

## First bounded implementation slice after research

### `R1-S1 run_greedy lookahead-path adoption`

Bounded slice:

- migrate `runtime_core.run_greedy` only
- keep the loop inline
- adopt the promoted lookahead helper path under an explicit core-owned gate/policy
- use the retained `decode.py` resolved-step helpers
- do not touch `general_path`
- do not touch `batch.scheduler`

Why this is the best first slice:

- it moves a real accepted benchmark win into the actual Layer 1 consumer
- it tests the benchmark/core split directly
- it preserves narrow blast radius
- it can be validated on the existing accepted-sensitive model set
- it does not recreate the rejected container shapes

## End state required before implementation resumes

Research is complete only when:

1. `ARCHITECTURE_NOTES.md` contains concrete maps for MLXs, `mlx_lm`, and MLX.
2. this file contains the ranked redesign thesis and first bounded slice.
3. `STATUS`, `TASKS`, `RISKS`, `DECISIONS`, `WORKLOG`, and `FREEZE` reflect the research mode.
