# Worklog

Append-only operational history. Keep it dense.

## 2026-04-18 — accepted baseline hardening summary

- Promoted sampled-pure short Layer 2 win.
- Promoted compile stabilization with prompt-aware decode eligibility `<=512`.
- Promoted chat optional `prompt_toolkit` boundary fix.
- Preserved accepted dirty baseline on `refactor/core-exec`.

## 2026-04-18 — acceptance and scheduler evidence

- Remote accepted baseline confirmed at `/tmp/mlxs_layer1_candidate_20260417_1`.
- AC2 scheduler authoritative baseline created:
  - `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`
  - `/tmp/mlxs_ac2_scheduler_baseline_ranked_20260418_1.json`
  - `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`
- Result: AC2 open; scheduler req/s behind at `256 0.91190x`, `2048 0.92751x`.

## 2026-04-18 — blocked and frozen families

- AC13 fair tuning ranked: `/tmp/mlxs_ac13_prefillstep_ranked_20260418_1.json`
- Bounded AC13 candidates exhausted:
  - `/tmp/mlxs_ac13_chunkcap_probe_local_20260418_2.json`
  - `/tmp/mlxs_ac13_streamhoist_probe_local_20260418_1.json`
- Result: AC13 blocked-for-now.

## 2026-04-18 to 2026-04-19 — Qwen compatibility and long decode

- Qwen compatibility candidate remote artifact:
  - `/tmp/mlxs_class_a_qwen15b_candidate_remote_20260418_1.json`
- Promoted Qwen accepted reruns:
  - `/tmp/mlxs_class_a_qwen15b_promoted_remote_20260418_2.json`
  - `/tmp/mlxs_class_a_qwen15b_promoted_remote_20260418_3.json`
- Long-decode supporting artifacts:
  - `/tmp/mlxs_qwen_long_decode_decomp_local_20260418_1.json`
  - `/tmp/mlxs_qwen_long_decode_substeps_local_20260418_1.json`
  - `/tmp/mlxs_qwen_long_prepared_compare_local_20260418_1.json`
- Result: Qwen benchmarkability restored; Qwen 2048 decode positive on accepted benchmark surface.

## 2026-04-19 — broadened AC1 set and Llama 3B

- Added `mlx-community/Llama-3.2-3B-Instruct-4bit` to remote host cache.
- Initial 3B baseline:
  - `/tmp/mlxs_class_a_llama32_3b_remote_20260418_1.json`
- 3B long-decode artifacts:
  - `/tmp/mlxs_ac1_llama32_3b_long_decode_decomp_remote_20260418_1.json`
  - `/tmp/mlxs_ac1_llama32_3b_long_prepared_probe_remote_20260418_1.json`
  - `/tmp/mlxs_ac1_llama3b_long_guardrails_confirm_20260418_1.json`
  - `/tmp/mlxs_class_a_llama32_3b_promoted_remote_20260418_1.json`
- Result: Llama 3B 2048 decode improved to positive on accepted benchmark surface.

## 2026-04-19 — current accepted breadth state

- Accepted references now:
  - Llama 1B
  - Qwen 1.5B
  - Llama 3B
- Broad-set rank:
  - `/tmp/mlxs_ac1_broadened_set_rank_20260419_1.json`
- Result: AC1 still open because of true performance gaps on the broadened accepted set.

## 2026-04-19 — M01.R7 resident aligned-cohort handoff attempted and rejected

- Slice:
  - resident prefill-to-first-decode handoff for aligned same-length cohorts
  - scheduler-internal only
  - no prompt-cache/product-surface contract change
- Implementation:
  - carried merged cohort cache and batched `y` across the prefill/decode boundary for one completion step
  - added focused scheduler tests for handoff residency, cancellation materialization, and finished-row cache ownership
- Focused correctness:
  - `6 passed` on `tests/unit/test_batch/test_scheduler.py tests/unit/test_product_surfaces/test_batched_serving.py`
- Local AC2 truth-first artifacts:
  - `/tmp/mlxs_m01_r7_worktree_local_ac2_20260419_1.json`
  - `/tmp/mlxs_m01_r7_main_local_ac2_20260419_1.json`
- Local compare vs accepted main baseline:
  - prompt `256`: `requests/s ~0.9829x`, `generated_tok/s ~0.9829x`, `p50 TTFT ~1.0417x`, `p95 completion ~1.0172x`
  - prompt `2048`: `requests/s ~1.0064x`, `generated_tok/s ~1.0064x`, `p50 TTFT ~1.0046x`, `p95 completion ~0.9954x`
  - parity: exact
- Outcome:
  - slice rejected locally
  - scheduler and test patches reverted in the redesign worktree
  - next path broadened to integrated batch-contract redesign ranking

## 2026-04-19 — file-backed ops kickoff

- Created `docs/execution/` operating system of record.
- Activated milestone `M01-performance-architecture-redesign`.
- Next step: rank redesign fronts from accepted evidence, then begin architecture archaeology on the shared decode/batch hot path.

## 2026-04-19 — M01 front ranking and first archaeology

- Read governing docs:
  - `docs/refactor/benchmark-protocol.md`
  - `docs/refactor/performance-core-spec.md`
  - `docs/refactor/canonical-refactor.md`
  - `docs/specs.md`
- Read live hot-path surfaces:
  - `src/mlxs/runtime_core/decode.py`
  - `src/mlxs/runtime_core/greedy.py`
  - `src/mlxs/runtime_core/prefill.py`
  - `src/mlxs/general_path/single_request.py`
  - `src/mlxs/batch/scheduler.py`
  - `benchmarks/mlxs_vs_mlx_lm/backends.py`
- Finding:
  - benchmark helper owns prepared-step/lookahead wins
  - `runtime_core.run_greedy()` still uses plain `decode_step`
  - `general_path` duplicates several prepared-step loops above Layer 1
  - `batch.scheduler` bypasses Layer 1 and owns its own prefill/decode/materialization/event-shaping loop
- Result:
  - first redesign front chosen: `resident decode state machine / hot-path unification`
  - second front remains downstream: `resident batch progression / AC2 throughput architecture`

## 2026-04-19 — M01 boundary readback

- `runtime_core` today exposes:
  - `CoreState`
  - `PreparedDecodeStep`
  - `PreparedNextLogits`
  - `CoreStepResult`
- Missing today:
  - one resident progression object/state machine that owns decode advancement across steps
- Current divergence:
  - `runtime_core.run_greedy()` still loops through plain `decode_step`
  - benchmark helper owns prepared-step/lookahead routing and model-specific long-prompt gates
  - `general_path` duplicates multiple prepared-step loops and Layer 2 enrichment around them
  - `batch.scheduler` bypasses Layer 1 and owns prefill/decode/materialization plus `TokenEvent` shaping directly
- Boundary conclusion:
  - Layer 1 redesign should own resident progression state, current/next token/logits handles, execution policy, termination, and cache ownership
  - text decode, stop-sequence text checks, logprob shaping, and `TokenEvent` construction stay above Layer 1

## 2026-04-19 — first redesign worktree opened

- Worktree: `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-resident-decode-state-machine`
- Branch: `codex/m01-resident-decode-state-machine`
- Baseline: current accepted dirty `refactor/core-exec` diff overlaid into the worktree, including `docs/execution/`
- Purpose: isolate the first architecture-redesign family without contaminating the accepted baseline

## 2026-04-19 — first resident decode slice attempted and rejected

- Slice:
  - introduce class-based `DecodeProgression`
  - migrate `runtime_core.run_greedy`
  - migrate benchmark helper
- Local correctness:
  - runtime-core + benchmark focused suite passed before remote try
- Local artifacts:
  - `/tmp/mlxs_m01_decode_progression_main_local_20260419_1.json`
  - `/tmp/mlxs_m01_decode_progression_worktree_local_20260419_1.json`
  - `/tmp/mlxs_m01_decode_progression_main_guardrails_local_20260419_1.json`
  - `/tmp/mlxs_m01_decode_progression_worktree_guardrails_local_20260419_1.json`
  - `/tmp/mlxs_m01_decode_progression_microprobe_local_20260419_1.json`
- Remote candidate:
  - `/tmp/mlxs_m01_resident_decode_candidate_20260419_1`
  - `/tmp/mlxs_m01_resident_decode_candidate_remote_20260419_1.json`
  - `/tmp/mlxs_m01_resident_decode_candidate_compare_20260419_1.json`
- Remote result:
  - Llama 1B `256 decode 0.91376x` vs accepted `0.99997x`
  - Llama 1B `2048 decode 1.07208x` vs accepted `1.07037x`
  - Qwen 1.5B `256 decode 0.86343x` vs accepted `1.02031x`
  - Qwen 1.5B `2048 decode 0.88304x` vs accepted `1.02585x`
- Outcome:
  - candidate reverted inside the worktree
  - next slice is lower-overhead resident progression state + module-level helpers

## 2026-04-19 — next slice narrowed

- First migrated consumer for the next slice: benchmark helper only.
- Surfaces intentionally untouched for the next slice:
  - `runtime_core.run_greedy`
  - `general_path.single_request`
  - `batch.scheduler`
- Minimum next state bundle:
  - `CoreState`
  - `CoreTerminationPolicy`
  - `CoreExecutionPolicy`
  - `StepFn`
  - `TokenSelector`
  - `use_lookahead`
  - resident current-step handle (`logits` or prepared token/logits)

## 2026-04-19 — low-overhead state slice attempted and rejected

- Slice:
  - data-only `DecodeProgressionState`
  - module-level `start_decode_progression` / `advance_decode_progression`
  - benchmark helper migrated only
- Focused validation:
  - `26 passed` on runtime-core + benchmark/helper boundary suite
- Debug note:
  - first local benchmark run was invalid because `generated_tokens` was unpacked into the wrong local name in the benchmark helper
  - fixed before classification
- Local accepted-sensitive guardrails:
  - `/tmp/mlxs_m01_low_overhead_local_guardrails_20260419_2.json`
  - baseline compare: `/tmp/mlxs_m01_low_overhead_main_guardrails_20260419_1.json`
- Local result vs main accepted helper:
  - Llama 1B `256 decode ~0.912x`
  - Llama 1B `2048 decode ~1.014x`
  - Qwen `256 decode ~0.885x`
  - Qwen `2048 decode ~0.876x`
- Outcome:
  - progression-state container removed from worktree
  - retained only neutral `runtime_core.decode` helper extraction
  - next slice is inline hot-loop helper reuse only

## 2026-04-19 — inline helper migration attempted and rejected

- Slice:
  - keep benchmark helper loop inline
  - replace wrapper calls with `_decode_step_with_resolved_step` and `_schedule_next_decode_step_with_resolved_step`
  - no container/state-object hot-loop shape
- Focused validation:
  - `22 passed` on runtime-core + benchmark/helper boundary suite
- Local artifacts:
  - `/tmp/mlxs_m01_inline_helpers_local_guardrails_20260419_1.json`
  - `/tmp/mlxs_m01_inline_helpers_worktree_qwen_confirm_20260419_1.json`
  - `/tmp/mlxs_m01_inline_helpers_main_qwen_confirm_20260419_1.json`
- Result:
  - Llama 1B `256/2048` improved locally
  - Qwen `256` was near-neutral/slightly better
  - Qwen `2048` remained worse than accepted main helper on confirm
- Outcome:
  - no remote run
  - helper migration rejected as mixed
  - retained helper extraction in `runtime_core.decode` as neutral groundwork

## 2026-04-19 — lookahead-branch-only helper migration promoted

- Phase 0-4:
  - target: benchmark-helper lookahead branch only
  - strategic value: last bounded branch-local leverage after three rejected broader shapes
  - divergence: only `schedule_next_decode_step(...)` remained a wrapper call inside the lookahead branch
  - no-patch branch-only probe:
    - `/tmp/mlxs_m01_lookahead_branch_probe_local_20260419_1.json`
    - signal: Qwen `256/2048` positive, Llama `2048` positive, Llama `256` near-neutral
- Patch:
  - benchmark helper only
  - swapped lookahead-branch wrapper call to `_schedule_next_decode_step_with_resolved_step`
  - pre-resolved `step_fn` once per trial
- Local validation:
  - `22 passed`
  - `/tmp/mlxs_m01_lookahead_branch_local_guardrails_20260419_1.json`
  - interleaved compare:
    - `/tmp/mlxs_m01_lookahead_branch_interleaved_compare_local_20260419_1.json`
    - all four accepted-sensitive cases neutral-to-better
- Remote accepted rerun:
  - `/tmp/mlxs_m01_lookahead_branch_promoted_remote_20260419_1.json`
  - compare:
    - `/tmp/mlxs_m01_lookahead_branch_promoted_compare_20260419_1.json`
  - accepted-remote result vs prior accepted:
    - Llama 1B `256 decode 1.00117x`
    - Llama 1B `2048 decode 0.99988x`
    - Qwen `256 decode 1.00298x`
    - Qwen `2048 decode 0.99967x`
- Outcome:
  - promoted as bounded structural win
  - next path is `runtime_core.run_greedy consumer ranking`

## 2026-04-19 — M01.6 initial consumer delta

- `runtime_core.run_greedy` still:
  - calls `decode_step(...)`
  - carries no lookahead gate
  - does not consume the retained resolved-step helpers
- Promoted benchmark helper now:
  - still keeps the loop inline
  - uses `_schedule_next_decode_step_with_resolved_step(...)` on the lookahead branch only
  - pre-resolves `step_fn` once per trial
- Side-by-side implication:
  - the benchmark surface now proves the helper-groundwork is safe on the lookahead branch
  - the next real architectural consumer is the canonical Layer 1 entrypoint, not `general_path` or `batch`

## 2026-04-19 — M01.R1 research mode activated

- Implementation hold:
  - no new performance slice until research completes
- Research files created:
  - `docs/execution/milestones/M01-performance-architecture-redesign/RESEARCH.md`
  - `docs/execution/milestones/M01-performance-architecture-redesign/ARCHITECTURE_NOTES.md`
- Research coverage:
  - MLXs hot path
  - `mlx_lm` hot path
  - MLX substrate constraints/opportunities
  - divergence map
  - bottleneck model
  - ranked redesign thesis + fallback + first bounded slice

## 2026-04-19 — M01.R1 research complete

- Research outputs:
  - `docs/execution/milestones/M01-performance-architecture-redesign/RESEARCH.md`
  - `docs/execution/milestones/M01-performance-architecture-redesign/ARCHITECTURE_NOTES.md`
- Outcome:
  - MLXs vs `mlx_lm` vs MLX maps written down
  - structural taxes separated from incidental taxes
  - ranked redesign thesis selected
  - fallback thesis selected
  - first bounded post-research implementation slice selected

## 2026-04-19 — M01.R1b MLX capability addendum complete

- Added milestone file:
  - `docs/execution/milestones/M01-performance-architecture-redesign/MLX_CAPABILITY_INVENTORY.md`
- Inventory covered:
  - lazy evaluation
  - eval / async_eval
  - streams / synchronization
  - compile / export
  - unified memory
  - memory controls
  - distributed communication
  - quantization / quantized kernels
  - selection primitives
  - specialized matmul / gather / block ops
- Outcome:
  - ranked redesign thesis unchanged
  - fallback thesis unchanged
  - next path remains `M01.R2 runtime_core.run_greedy lookahead-path adoption`

## 2026-04-19 — M01.R2 framing

- Target:
  - `runtime_core.run_greedy` only
- Why this consumer first:
  - it is the real canonical Layer 1 entrypoint
  - benchmark-helper lookahead win is already promoted and safe
  - moving that win into core closes the benchmark/core split directly
- Bounded slice:
  - adopt generic short-prompt lookahead only
  - use retained `decode.py` helpers
  - no model-specific long-prompt gate in `run_greedy` yet
  - keep `general_path` and `batch` untouched

## 2026-04-19 — M01.R2 run_greedy lookahead-path adoption promoted

- Patch:
  - `src/mlxs/runtime_core/greedy.py`
- Exact change:
  - adopt generic short-prompt lookahead (`<=512`) in `run_greedy`
  - consume retained `decode.py` helpers
  - keep non-lookahead path on `decode_step`
- Local validation:
  - `16 passed` focused runtime-core suite
  - interleaved compare:
    - `/tmp/mlxs_m01_r2_run_greedy_interleaved_compare_local_20260419_1.json`
  - local result:
    - Llama 1B `256 decode 1.0968x`
    - Llama 1B `2048 decode 1.0363x`
    - Qwen `256 decode 1.0007x`
    - Qwen `2048 decode 1.0464x`
- Remote authoritative compare:
  - `/tmp/mlxs_m01_r2_run_greedy_candidate_compare_remote_20260419_1.json`
  - remote result:
    - Llama 1B `256 decode 1.0074x`
    - Llama 1B `2048 decode 1.0001x`
    - Qwen `256 decode 1.0073x`
    - Qwen `2048 decode 0.9984x`
- Outcome:
  - promoted as bounded structural win
  - next redesign work should move to the next adjacent core consumer/path

## 2026-04-19 — M01.R3 consumer ranking

- Compared next consumer candidates:
  - `general_path single_request`
  - `batch.scheduler`
- Ranking basis:
  - progression ownership
  - lookahead/helper reuse opportunity
  - shaping boundary
  - blast radius
  - validation surface
  - acceptance leverage
- Result:
  - `general_path single_request` ranked ahead of `batch.scheduler`
  - reason: it already sits on top of Layer 1 contracts and duplicates prepared-step progression directly, while batch still combines resident-state rebuild, batching policy, and `TokenEvent` shaping in one hot loop
- First bounded slice:
  - migrate `use_prepared_step_enriched_path`
  - migrate `use_sampled_prepared_step_path`
  - leave `use_processor_prepared_step_path` untouched
  - leave baseline decode path untouched

## 2026-04-19 — broad general_path prepared-step slice attempted and rejected

- Patch:
  - `src/mlxs/general_path/single_request.py`
- Scope:
  - prepared-step enriched path
  - prepared-step sampled path
  - processor path untouched
  - baseline decode path untouched
- Local validation:
  - `29 passed` on focused general-path + boundary + runtime-core suite
  - interleaved compare:
    - `/tmp/mlxs_m01_r3_general_path_interleaved_compare_local_20260419_1.json`
  - focused Qwen short-logprobs confirm:
    - `/tmp/mlxs_m01_r3_qwen_short_logprobs_confirm_20260419_1.json`
- Result:
  - Llama short logprobs improved
  - Llama short sampled stayed neutral
  - Qwen long top-logprobs improved
  - long plain logprobs guardrail stayed neutral
  - Qwen short plain logprobs regressed on focused confirm
- Outcome:
  - slice rejected locally before remote budget
  - worktree and accepted baselines reverted to pre-slice state
  - next path narrowed to enriched top-logprobs-only

## 2026-04-19 — narrowed top-logprobs-only general_path slice attempted and rejected

- Slice:
  - `general_path` enriched `top_logprobs` subpath only
  - sampled, processor, and baseline decode subpaths untouched
- Focused validation:
  - `30 passed` on focused general-path + boundary + runtime-core suite
  - interleaved compare:
    - `/tmp/mlxs_m01_r4_toplogprobs_interleaved_compare_local_20260419_1.json`
- Result:
  - Llama short `top_logprobs` near-neutral/slightly negative
  - Qwen short `top_logprobs` near-neutral/slightly negative on e2e
  - Qwen long `top_logprobs` near-neutral/slightly negative
  - short plain-logprobs guardrail stayed neutral
- Outcome:
  - slice rejected locally before remote budget
  - worktree reverted to accepted `general_path` baseline
  - `general_path` follow-up slices exhausted for this phase

## 2026-04-19 — M01.R5 batch architecture framing and divergence read

- Accepted AC2 state re-read:
  - `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`
  - `/tmp/mlxs_ac2_scheduler_baseline_ranked_20260418_1.json`
  - `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`
- Current accepted AC2 picture:
  - `256 req/s 0.91190x`
  - `2048 req/s 0.92751x`
  - short prompt cost dominated by `decode_active`
  - long prompt split between `prefill_cohort` and `decode_active`
- Side-by-side batch architecture read:
  - `BatchScheduler` still owns:
    - prompt prefill
    - grouped cache merge/scatter
    - active decode loop
    - sampling
    - `y.item()`
    - tokenizer decode
    - `TokenEvent` creation
  - `mlx_lm.BatchGenerator` keeps a resident `Batch` object with:
    - `y`
    - `logprobs`
    - `cache`
    - `tokens`
    - `uids`
    - `num_tokens`
    - `max_tokens`
    - `samplers`
    - `logits_processors`
    - in-place `filter` / `extend`
    - `mx.async_eval(batch.y, batch.logprobs, batch.tokens)`
- Result:
  - next structural AC2 front is correctly `resident batch progression / AC2 throughput architecture`
  - scheduler micro-fix churn remains closed

## 2026-04-19 — M01.R5 candidate ranking

- Candidate 1:
  - resident active completion-batch object
  - focus: decode progression ownership, resident `y/logprobs/tokens/cache`, and in-place active filtering
- Candidate 2:
  - prefill/prompt-processing unification around a resident batch object
  - focus: prompt checkpoint, finalize, imported-cache merge, and prompt-processing continuity
- Ranking:
  - Candidate 1 first
  - Candidate 2 fallback
- Reason:
  - accepted AC2 `256` is decode-dominated
  - accepted AC2 `2048` still has a large decode share even though prefill is also material
  - Candidate 1 is the smaller bounded architecture slice

## 2026-04-19 — M01.R6 batch slice framing

- First batch slice:
  - `resident active completion-batch object`
- Persistent fields in scope:
  - ordered active entries / request-sequence refs
  - merged resident batched KV cache
  - resident current token batch `y`
  - minimal decode counters needed for completion progression
- Explicitly out of scope:
  - prompt processing / prompt checkpointing
  - prompt-cache prepare/finalize integration
  - host/product stream plumbing
  - broad event/logprob redesign
  - queue admission semantics
- Why first:
  - decode dominates accepted AC2 `256`
  - decode remains a large share of accepted AC2 `2048`
  - this is the narrowest slice that can test resident-batch ownership

## 2026-04-19 — M01.R6 resident completion-batch probe rejected

- No-patch truth-first probe:
  - `/tmp/mlxs_m01_r6_completion_batch_probe_local_20260419_1.json`
- Canonical AC2 surface:
  - `N=4`
  - `max_tokens=128`
  - prompt `256`
  - prompt `2048`
  - greedy / no-logprobs / no-processors
- Result:
  - `256 requests/s ~0.968x`
  - `2048 requests/s ~0.917x`
  - TTFT/completion worse on both
  - parity false on both
- Outcome:
  - slice rejected before any repo patch
  - next path is the fallback prompt-processing/prefill unification around a resident batch object

## 2026-04-19 — M01.R7 initial fallback framing

- Current best bounded target:
  - resident prefill-to-completion handoff for aligned cohorts
- Why:
  - `_prefill_cohort` still builds a merged cache and immediately scatters it back into per-sequence caches
  - accepted AC2 `2048` still has a large prefill share
  - this is narrower than a full prompt-processing rewrite and does not require product-surface changes

## 2026-04-19 — M01.R8 re-ranking complete, R9 implementation opened

- Fresh redesign worktree opened:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r9-integrated-fast-batch`
  - branch `codex/m01-r9-integrated-fast-batch`
- Accepted dirty baseline overlaid into the new worktree; `docs/execution/` copied in as system of record.
- Re-ranking conclusion:
  - the missing batch substrate is now explicit
  - MLXs has no batch-aware cache/runtime contract at all
  - `mlx_lm` batch throughput advantage depends on a resident `Batch` owner plus batch-aware cache ops (`filter`, `extend`, `extract`, `merge`)
- Result:
  - `R8` closed as ranking complete
  - active implementation path is `R9 integrated resident fast-batch contract`
  - scope locked to canonical AC2 fast path only

## 2026-04-19 — M01.R9 integrated resident fast-batch contract attempted and rejected

- Integrated redesign slice:
  - new private resident batch runtime in `src/mlxs/batch/active_batch.py`
  - `BatchScheduler` refactored into orchestration over:
    - resident canonical fast batch
    - accepted legacy fallback path
  - retained neutral tooling:
    - `benchmarks/mlxs_vs_mlx_lm/scheduler_ac2_probe.py`
- Focused correctness:
  - `5 passed` on `tests/unit/test_batch/test_scheduler.py`
  - `4 passed` on `tests/unit/test_product_surfaces/test_batched_serving.py`
- Local canonical AC2 compare:
  - worktree artifact: `/tmp/mlxs_m01_r9_worktree_local_ac2_20260419_1.json`
  - accepted-main artifact: `/tmp/mlxs_m01_r9_main_local_ac2_20260419_1.json`
  - result:
    - prompt `256`: `requests/s ~1.2629x`, `generated_tok/s ~1.2629x`, `TTFT ~0.9787x`, `p95 completion ~0.7918x`
    - prompt `2048`: `requests/s ~1.4930x`, `generated_tok/s ~1.4930x`, `TTFT ~0.9627x`, `p95 completion ~0.6701x`
    - parity: exact
- Remote authoritative compare:
  - remote accepted tree was stale on the batch surface, so two fresh remote mirrors were used:
    - control: `/tmp/mlxs_m01_r9_control_local_20260419_1`
    - candidate: `/tmp/mlxs_m01_r9_candidate_local_20260419_1`
  - candidate artifact: `/tmp/mlxs_m01_r9_candidate_remote_ac2_20260419_2.json`
  - control artifact: `/tmp/mlxs_m01_r9_control_remote_ac2_20260419_2.json`
  - result:
    - prompt `256`: `requests/s ~2.2798x`, `generated_tok/s ~2.2798x`, `TTFT ~0.4466x`, `p95 completion ~0.4386x`
    - prompt `2048`: `requests/s ~0.5995x`, `generated_tok/s ~0.5995x`, `TTFT ~2.0935x`, `p95 completion ~1.6682x`
    - parity: exact
- Outcome:
  - runtime redesign rejected as non-promotable
  - runtime code reverted in the redesign worktree
  - scheduler-level AC2 probe retained as neutral tooling
  - next path moves to shared Layer 1 + Layer 3 batch-capable progression ranking

## 2026-04-19 — M01.R10 diagnosis and first integrated slice

- Fresh redesign worktree opened:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r10-shared-batch-progression`
  - branch `codex/m01-r10-shared-batch-progression`
- Post-`R9` diagnosis:
  - short-prompt remote win came from removing Layer 3 reconstruction cost
  - long-prompt remote loss showed that Layer 3-only residency still leaves the wrong owner for prompt prefill, current/next token progression, and materialization
- Ranked thesis:
  - shared Layer 1 batch-capable progression kernel consumed by Layer 3 orchestration
- First integrated slice:
  - aligned plain-KV shared batch progression kernel
  - Layer 1 owns batched prefill, prepared-step scheduling, materialization, and row filter/extract cache ops
  - Layer 3 owns queueing, row metadata, stop/event shaping, and legacy fallback

## 2026-04-19 — M01.R10 shared batch progression promoted

- Implemented:
  - `src/mlxs/runtime_core/batch_progression.py`
  - `src/mlxs/batch/fast_batch.py`
  - `src/mlxs/batch/scheduler.py` orchestration split
  - repo-native AC2 probe `benchmarks/mlxs_vs_mlx_lm/scheduler_ac2_probe.py`
- Focused correctness:
  - local main after promotion: `12 passed` on runtime-core batch progression + batch scheduler + batched serving suites
- Local AC2 compare:
  - `/tmp/mlxs_m01_r10_worktree_local_ac2_20260419_1.json`
  - `/tmp/mlxs_m01_r10_main_local_ac2_20260419_1.json`
  - result:
    - prompt `256`: `requests/s ~1.2260x`, `generated_tok/s ~1.2260x`, `TTFT ~0.9302x`, `p95 completion ~0.8156x`
    - prompt `2048`: `requests/s ~1.4604x`, `generated_tok/s ~1.4604x`, `TTFT ~0.9790x`, `p95 completion ~0.6852x`
    - parity: exact
- Remote authoritative compare:
  - used fresh remote mirrors again:
    - control: `/tmp/mlxs_m01_r10_control_local_20260419_1`
    - candidate: `/tmp/mlxs_m01_r10_candidate_local_20260419_1`
  - artifacts:
    - `/tmp/mlxs_m01_r10_candidate_remote_ac2_20260419_1.json`
    - `/tmp/mlxs_m01_r10_control_remote_ac2_20260419_1.json`
  - result:
    - prompt `256`: `requests/s ~1.1416x`, `generated_tok/s ~1.1416x`, `TTFT ~0.9736x`, `p95 completion ~0.8759x`
    - prompt `2048`: `requests/s ~1.1875x`, `generated_tok/s ~1.1875x`, `TTFT ~1.1090x`, `p95 completion ~0.8421x`
    - parity: exact
- Outcome:
  - `R10` promoted into accepted local baseline and accepted remote tree
  - next path is post-promotion acceptance rerank, not immediate new implementation
