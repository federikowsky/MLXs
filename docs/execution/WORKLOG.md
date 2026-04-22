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

## 2026-04-19 — durable accepted-baseline checkpoint committed

- Commit: `fcb69de`
- Type: milestone checkpoint commit
- Captures:
  - accepted promoted baseline through `R10`
  - promoted shared Layer 1 + Layer 3 batch progression kernel
  - repo-native scheduler AC2 probe
  - accepted file-backed execution system through `R11` handoff
- Excludes:
  - rejected redesign worktree code
  - remote mirror trees
  - reverted experimental variants

## 2026-04-19 — M01.R11 acceptance rerank from durable checkpoint

- Rerank base:
  - durable accepted checkpoint `fcb69de`
  - docs/hash follow-up `74b8843`
- Normalization result:
  - `AC1`: open and unchanged by `R10`
  - `AC13`: blocked and unchanged by `R10`
  - `AC2`: strongest next open target
- Direct current accepted-surface AC2 rerun on the approved remote host:
  - accepted MLXs artifact reused from promoted `R10` remote run:
    - `/tmp/mlxs_m01_r10_candidate_remote_ac2_20260419_1.json`
  - fresh direct `mlx_lm` baseline:
    - `/tmp/mlxs_r11_mlx_lm_remote_ac2_20260419_2.json`
  - current ratios:
    - prompt `256`: `requests/s ~0.58268x`, `generated_tok/s ~0.58268x`, `TTFT ~1.4592x`, `p95 completion ~1.7162x`
    - prompt `2048`: `requests/s ~0.61619x`, `generated_tok/s ~0.61619x`, `TTFT ~1.4517x`, `p95 completion ~1.6229x`
- Outcome:
  - historical pre-`R10` AC2 ratios are no longer sufficient as the normalized acceptance ledger
  - next active path becomes `R12 AC2 accepted-surface reconciliation and closure rerun`

## 2026-04-19 — M01.R12 AC2 accepted-surface reconciliation complete

- Reconciliation result:
  - the current direct accepted-surface AC2 rerun is the right surface for the promoted baseline
  - the old pre-`R10` AC2 ledger is superseded for current acceptance ranking
- Reason:
  - direct current `mlx_lm` rerun on the approved remote host stayed close to the old authoritative comparator
  - the promoted accepted MLXs AC2 surface remained far below it
- Current accepted-surface AC2 ratios:
  - prompt `256`: `requests/s ~0.58268x`, `generated_tok/s ~0.58268x`, `TTFT ~1.4592x`, `p95 completion ~1.7162x`
  - prompt `2048`: `requests/s ~0.61619x`, `generated_tok/s ~0.61619x`, `TTFT ~1.4517x`, `p95 completion ~1.6229x`
- Outcome:
  - AC2 remains the strongest open target
  - next path becomes `R13 AC2 accepted-surface provenance audit and ledger rewrite`

## 2026-04-19 — M01.R15 post-R14 target rerank complete

- Governing accepted evidence re-read:
  - repo/file-backed system in `docs/execution/`
  - accepted AC1 artifacts:
    - `/tmp/mlxs_class_a_accepted_remote_20260418_1.json`
    - `/tmp/mlxs_class_a_qwen15b_promoted_remote_20260418_3.json`
    - `/tmp/mlxs_class_a_llama32_3b_promoted_remote_20260418_1.json`
  - repaired AC2 direct comparator:
    - `/tmp/mlxs_ac2_direct_compare_candidate_remote_20260419_2.json`
  - surviving AC13 authority:
    - file-backed `best eager prefill 1.02808x < 1.05x`
- Normalization result:
  - `AC1`: open and strongest
  - `AC2`: no longer dominant; prompt `256 ~0.99661x`, prompt `2048 ~1.09404x`
  - `AC13`: blocked and unchanged
- Outcome:
  - stale `R11/R12/R13` AC2 collapse ledger is superseded
  - next path moves to `R16 AC1 weakest-case decomposition and benchmark/core convergence ranking`

## 2026-04-19 — M01.R16 AC1 weakest-case decomposition complete

- Local truth-first helper/core split artifact:
  - `/tmp/mlxs_m01_r16_local_core_split_20260419_1.json`
- Local result:
  - `Llama-3.2-1B 256`: core/helper decode ratio `~0.8212`
  - `Llama-3.2-1B 2048`: core/helper decode ratio `~1.0277`
  - `Qwen2.5-1.5B 256`: core/helper decode ratio `~0.7911`
  - generated token counts matched in all local cases
- Remote truth-first Class A reruns on the accepted tree:
  - `/tmp/mlxs_m01_r16_llama1b_remote_class_a_20260419_1.json`
  - `/tmp/mlxs_m01_r16_qwen256_remote_class_a_20260419_1.json`
  - `/tmp/mlxs_m01_r16_llama3b256_remote_class_a_20260419_1.json`
- Remote rerun role:
  - used as truth-first guardrails only
  - not promoted replacements for the accepted AC1 ledger
- Interpretation:
  - the benchmark/core split is still materially open on short-prompt weakest-case surfaces
  - the long-prompt `Llama-3.2-1B 2048` case no longer points to the same gap
- Outcome:
  - next active path becomes `R17 AC1 short-prompt helper/core convergence slice definition`

## 2026-04-19 — M01.R17 exact first slice isolated

- No-patch local probe artifact:
  - `/tmp/mlxs_m01_r17_no_group_materialize_probe_local_20260419_1.json`
- Hypothesis tested:
  - `run_greedy` short-prompt path currently passes `next_prepared` into `materialize_prepared_step()`
  - benchmark helper schedules `next_prepared` but does not group that eval at materialization time
- Local result:
  - `Llama-3.2-1B 256`: no-group/current decode ratio `~1.10696x`
  - `Qwen2.5-1.5B 256`: no-group/current decode ratio `~1.09965x`
  - token counts and first-token prefixes matched on both cases
- Outcome:
  - `R17` is no longer a generic convergence placeholder
  - exact next slice is `AC1 short-prompt materialization contract alignment`

## 2026-04-19 — M01.R17 isolated runtime worktree opened

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r17-ac1-short-prompt-alignment`
- Branch:
  - `codex/m01-r17-ac1-short-prompt-alignment`
- Baseline:
  - commit `74b8843`
- Purpose:
  - implement the narrow `R17` short-prompt materialization-alignment candidate without contaminating the accepted baseline

## 2026-04-19 — M01.R17 candidate rejected and reverted

- Implemented candidate:
  - short-prompt `run_greedy` materialization alignment
  - specifically: do not pass `next_prepared` into `materialize_prepared_step()` on the short-prompt path
  - long-prompt baseline path untouched
- Focused correctness in the isolated worktree:
  - `6 passed` on `tests/unit/test_runtime_core/test_greedy.py tests/unit/test_benchmarks/test_backends.py`
  - `ruff check` passed on the touched runtime files
- Local candidate-vs-baseline artifact:
  - `/tmp/mlxs_m01_r17_local_compare_20260419_1.json`
  - result:
    - `Llama-3.2-1B 256 candidate/baseline decode ratio ~1.26068x`
    - `Qwen2.5-1.5B 256 candidate/baseline decode ratio ~1.24726x`
    - `Llama-3.2-1B 2048 candidate/baseline decode ratio ~0.93617x`
    - token counts and token prefixes exact on all local cases
- Remote candidate tree:
  - `/tmp/mlxs_m01_r17_candidate_20260419_1`
- Remote candidate artifacts:
  - `/tmp/mlxs_m01_r17_llama1b_candidate_remote_20260419_1.json`
  - `/tmp/mlxs_m01_r17_qwen256_candidate_remote_20260419_1.json`
  - `/tmp/mlxs_m01_r17_llama3b256_candidate_remote_20260419_1.json`
- Remote result:
  - `Llama-3.2-1B 256 eager decode ~0.99920x`
  - `Llama-3.2-1B 2048 eager decode ~1.06966x`
  - `Qwen2.5-1.5B 256 eager decode ~0.81891x`
  - `Llama-3.2-3B 256 eager decode ~1.96240x` on reduced-run truth-first guardrail
- Classification:
  - non-promotable
  - the decisive issue is not just mixed remote ratios; it is that a `run_greedy`-only slice does not directly move the current accepted AC1 helper benchmark surface
- Outcome:
  - candidate reverted cleanly in the `R17` worktree
  - next path reranked to `R18 AC1 shared short-prompt helper/core convergence slice definition`

## 2026-04-19 — M01.R18 isolated runtime worktree opened

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r18-ac1-shared-short-prompt`
- Branch:
  - `codex/m01-r18-ac1-shared-short-prompt`
- Baseline:
  - commit `74b8843`
- Purpose:
  - define and implement the first shared short-prompt helper/core convergence slice without reusing the rejected `R17` branch

## 2026-04-19 — M01.R18 candidate rejected locally and reverted

- Shared slice implemented in isolated worktree:
  - one shared short-prompt prepared-step loop in `runtime_core.greedy`
  - consumed by both `runtime_core.run_greedy` and the benchmark helper short-prompt path
  - long-prompt helper path and long-prompt core path left unchanged
- Focused correctness:
  - `8 passed` on `tests/unit/test_runtime_core/test_greedy.py tests/unit/test_benchmarks/test_backends.py`
  - `ruff check` passed on the touched files
- Local candidate surface artifact:
  - `/tmp/mlxs_m01_r18_local_surface_20260419_candidate.json`
- Local accepted-main surface artifact:
  - `/tmp/mlxs_m01_r18_local_surface_20260419_baseline.json`
- Local compare vs accepted main baseline:
  - `Llama-3.2-1B 256`: helper decode ratio `~0.67583x`, core decode ratio `~0.52537x`
  - `Llama-3.2-1B 2048`: helper decode ratio `~1.24697x`, core decode ratio `~1.00114x`
  - `Qwen2.5-1.5B 256`: helper decode ratio `~2.23903x`, core decode ratio `~1.54086x`
  - local `Llama-3.2-3B 256` accepted snapshot unavailable
  - token counts and token prefixes exact on all local cases
- Classification:
  - non-promotable before remote
  - weakest accepted case regressed too sharply to justify remote budget
  - failure mode points to per-token generator abstraction overhead, not correctness drift
- Outcome:
  - candidate reverted cleanly in the `R18` worktree
  - next path reranked to `R19 AC1 low-overhead inline short-prompt convergence ranking`

## 2026-04-19 — M01.R19 isolated runtime worktree opened

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r19-ac1-inline-short-prompt`
- Branch:
  - `codex/m01-r19-ac1-inline-short-prompt`
- Baseline:
  - commit `74b8843`
- Purpose:
  - rank and define the first lower-overhead shared short-prompt convergence slice without reusing the rejected `R18` branch

## 2026-04-20 — M01.R19 candidate rejected and path reranked

- Lower-overhead shared slice implemented in isolated worktree:
  - shared per-iteration short-prompt helper in `runtime_core.decode`
  - consumed by both `runtime_core.run_greedy` and the benchmark helper short-prompt path
  - no shared generator loop
  - long-prompt paths unchanged
- Focused correctness:
  - `22 passed` on `tests/unit/test_runtime_core/test_decode.py tests/unit/test_runtime_core/test_greedy.py tests/unit/test_benchmarks/test_backends.py`
  - `ruff check` passed on the touched files
- Local candidate surface artifact:
  - `/tmp/mlxs_m01_r19_local_surface_20260420_candidate.json`
- Local compare vs accepted-main surface:
  - `Llama-3.2-1B 256`: helper decode ratio `~1.15074x`, core decode ratio `~1.28201x`
  - `Llama-3.2-1B 2048`: helper decode ratio `~1.46739x`, core decode ratio `~1.24767x`
  - `Qwen2.5-1.5B 256`: helper decode ratio `~2.24291x`, core decode ratio `~1.49742x`
  - token counts and token prefixes exact on all local cases
- Remote control/candidate trees:
  - control: `/tmp/mlxs_m01_r19_control_20260420_1`
  - candidate: `/tmp/mlxs_m01_r19_candidate_20260420_1`
- Remote decisive artifacts:
  - `/tmp/mlxs_m01_r19_llama1b_control_remote_20260420_1.json`
  - `/tmp/mlxs_m01_r19_llama1b_candidate_remote_20260420_1.json`
- Remote result on the weakest accepted case:
  - control `Llama-3.2-1B 256 eager decode ~1.00214x`
  - candidate `Llama-3.2-1B 256 eager decode ~0.99967x`
  - `2048` stayed near-neutral (`~1.06997x` control vs `~1.06784x` candidate)
- Classification:
  - non-promotable
  - even the lower-overhead convergence family still fails to move the decisive authoritative weakest-case surface
- Outcome:
  - candidate is rejected
  - next path reranked to `R20 AC1 weakest-case helper-surface hot-loop decomposition and rerank`

## 2026-04-20 — M01.R20 isolated runtime worktree opened

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r20-ac1-helper-hotloop`
- Branch:
  - `codex/m01-r20-ac1-helper-hotloop`
- Baseline:
  - commit `74b8843`
- Purpose:
  - decompose the accepted weakest-case helper surface in isolation before choosing another AC1 implementation family

## 2026-04-20 — M01.R22 isolated runtime worktree opened

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r22-ac1-resident-lookahead-kernel`
- Branch:
  - `codex/m01-r22-ac1-resident-lookahead-kernel`
- Baseline:
  - commit `fcb69de`
- Purpose:
  - test a lower-level resident token-lookahead helper contract under the `R21` redesign front without contaminating the accepted baseline

## 2026-04-20 — M01.R22 local validation complete

- Implemented candidate in the isolated worktree:
  - added token-only Layer 1 lookahead helpers in `runtime_core.decode`
  - moved the benchmark helper short-prompt lookahead path onto a resident token loop
  - left the long-prompt non-lookahead baseline branch intact
- Focused correctness:
  - `18 passed` on `tests/unit/test_runtime_core/test_decode.py tests/unit/test_runtime_core/test_greedy.py tests/unit/test_benchmarks/test_backends.py`
  - `ruff check` passed on the touched files
- Local artifacts:
  - initial shortest-case readback: `/tmp/mlxs_m01_r22_baseline_local_20260420_1.json`
  - local control mirror: `/tmp/mlxs_m01_r22_control_local_20260420_1.json`
  - local candidate: `/tmp/mlxs_m01_r22_candidate_local_20260420_2.json`
- Local result vs control:
  - `Llama-3.2-1B 256 candidate/control decode ratio ~1.00480x`
  - `Llama-3.2-1B 2048 candidate/control decode ratio ~0.98952x`
  - TTFT moved slightly positive, but decode throughput signal was too small/mixed to justify any claim by itself
- Outcome:
  - local evidence was not strong enough for promotion
  - remote authoritative validation was still run because the user explicitly required remote testing

## 2026-04-20 — M01.R22 remote validation complete and candidate rejected

- Remote mirrors:
  - control: `/tmp/mlxs_m01_r22_control_20260420_1`
  - candidate: `/tmp/mlxs_m01_r22_candidate_20260420_1`
- Remote sync verification:
  - synced bounded diff only:
    - `benchmarks/mlxs_vs_mlx_lm/backends.py`
    - `src/mlxs/runtime_core/decode.py`
    - `tests/unit/test_runtime_core/test_decode.py`
  - local and remote `sha256` matched on all synced files before benchmarking
- Remote execution notes:
  - approved host/interpreter used:
    - host `llm@169.254.225.109`
    - interpreter `/Users/Shared/mlx-cluster-venv/bin/python`
  - remote interpreter does not have `pytest`, so the remote validation surface for this slice was benchmark-authoritative only
- Remote artifacts:
  - `/tmp/mlxs_m01_r22_control_remote_20260420_1.json`
  - `/tmp/mlxs_m01_r22_candidate_remote_20260420_1.json`
- Remote result vs control:
  - `Llama-3.2-1B 256 candidate/control MLXs decode ratio ~0.99904x`
  - `Llama-3.2-1B 2048 candidate/control MLXs decode ratio ~0.99686x`
  - remote candidate `mlxs/mlx_lm` ratios looked slightly better, but the paired `mlx_lm` medians drifted downward too:
    - `256`: `~191.75 -> ~191.12 tok/s`
    - `2048`: `~149.50 -> ~143.42 tok/s`
- Classification:
  - non-promotable
  - helper token-choreography simplification is not enough to move the decisive weakest-case surface
- Outcome:
  - `R22` is rejected
  - next active path reranked to `M01.R23 AC1 model/cache forward-step substrate parity audit`

## 2026-04-20 — M01.R23 isolated runtime worktree opened

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r23-ac1-forward-step-substrate`
- Branch:
  - `codex/m01-r23-ac1-forward-step-substrate`
- Baseline:
  - commit `fcb69de`
- Purpose:
  - audit the prompt-tail to first-decode substrate below the exhausted helper-only redesign family

## 2026-04-20 — M01.R23 audit artifact and first no-patch readback recorded

- Live audit artifact:
  - `docs/execution/milestones/M01-performance-architecture-redesign/R23_SUBSTRATE_AUDIT.md`
- Current substrate mapping recorded:
  - MLXs decisive path uses `mlxs.models.llama.Model` + `mlxs.cache.kv.KVCache`
  - `mlx_lm` decisive path uses `mlx_lm.models.llama.Model` + `mlx_lm.models.cache.KVCache`
- Early no-patch local readback on `Llama-3.2-1B 256`:
  - simple steady-state cached forward medians stayed near parity:
    - MLXs `~0.01273s`
    - `mlx_lm ~0.01267s`
  - simple first-decode-after-prefill read stayed directionally worse for MLXs:
    - MLXs `~0.01480s`
    - `mlx_lm ~0.00989s`
  - isolated synthetic first-call `KVCache.update_and_fetch()` asymmetry did not survive order reversal and is not yet actionable
- Interpretation:
  - broad steady-state model forward is not the first patch target
  - the first decisive substrate target is prompt-tail to first-decode handoff

## 2026-04-20 — M01.R23 local + remote substrate probes complete and path closed

- Local artifact:
  - `/tmp/mlxs_m01_r23_forward_step_probe_local_20260420_1.json`
- Remote artifact:
  - `/tmp/mlxs_m01_r23_forward_step_probe_remote_20260420_1.json`
- Remote validation setup:
  - remote mirror: `/tmp/mlxs_m01_r23_probe_remote_20260420_1`
  - synced bounded diff only:
    - `benchmarks/mlxs_vs_mlx_lm/r23_forward_step_probe.py`
  - local and remote `sha256` matched before the run
- Stable result across local and remote:
  - current MLXs total prompt-tail / first-decode boundary was already better than comparable `mlx_lm`
  - the first no-patch integrated boundary probe was materially worse than current MLXs
  - isolated `first_decode_forward` stayed slower in MLXs, but that did not translate into a better bounded redesign candidate for the full boundary
- Remote shape detail:
  - current MLXs `tail_to_next_ready ~0.00775s`
  - comparable `mlx_lm tail_to_next_ready ~0.00882s`
  - integrated probe/current ratio `~1.92319x`
  - current MLXs cache stayed pre-reserved at `512`, while comparable `mlx_lm` grew `256 -> 512` on the first decode step
- Classification:
  - negative audit
  - no bounded single-request substrate patch justified
- Outcome:
  - close `R23`
  - pivot to `R24 AC2 batch-first resident progression redesign rerank`

## 2026-04-20 — M01.R24 isolated runtime worktree opened

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r24-ac2-batch-first-redesign`
- Branch:
  - `codex/m01-r24-ac2-batch-first-redesign`
- Baseline:
  - commit `fcb69de`
- Purpose:
  - restart the batch-first redesign fallback from the repaired direct AC2 comparator and the now-closed `R23` single-request front

## 2026-04-20 — M01.R24 current batch divergence map rebuilt

- New live artifact:
  - `docs/execution/milestones/M01-performance-architecture-redesign/R24_BATCH_DIVERGENCE_MAP.md`
- Authoritative current AC2 surface locked:
  - `/tmp/mlxs_ac2_direct_compare_candidate_remote_20260419_2.json`
  - prompt `256 ~0.99661x`
  - prompt `2048 ~1.09404x`
  - exact parity on both prompts
- Old AC2 beliefs explicitly superseded:
  - stale post-`R10` collapse ledger is not the current acceptance ledger
  - prompt `2048` is no longer the dominant blocker
  - old scheduler micro-fix ranking is not reusable by inertia
- Current artifact replay used:
  - scheduler baseline/decomposition remote `2026-04-18`
  - `R9` remote control/candidate AC2 artifacts
  - `R10` remote control/candidate AC2 artifacts
  - current local direct comparator and scheduler stream probe snapshots
- Current batch readback:
  - `BatchScheduler` still splits between narrow `_SharedFastBatch` and broad legacy scheduler ownership
  - `mlx_lm` still uses resident `PromptProcessingBatch` + `GenerationBatch`
  - current dominant remaining taxes are architectural:
    - prompt-processing / generation handoff
    - extendable active-batch ownership
    - cache substrate breadth for canonical residency
  - event/materialization placement is not reranked as the main thesis
- Outcome:
  - ranked thesis chosen: canonical resident prompt-batch + extendable active-batch ownership over shared Layer 1 batch progression
  - fallback thesis chosen: bounded resident prompt-prefill handoff into an extendable fast batch owner
  - next active slice becomes `R24-S1 current fast-path prompt/handoff decomposition`

## 2026-04-20 — M01.R24-S1 current fast-path prompt/handoff decomposition complete

- Local artifact:
  - `/tmp/mlxs_m01_r24_fast_batch_decomp_local_20260420_1.json`
- Remote artifact:
  - `/tmp/mlxs_m01_r24_fast_batch_decomp_remote_20260420_1.json`
- Probe semantics:
  - current MLXs canonical fast path via `_SharedFastBatch`
  - comparable upstream prompt/generation path via `PromptProcessingBatch` + `GenerationBatch`
- Stable result:
  - activation is already competitive or better for MLXs
  - the first active-batch generation step is the stable slower component
- Local summary:
  - activation ratio `MLXs/upstream ~0.97728x`
  - first-step ratio `~1.67253x`
  - total ratio `~1.04600x`
- Remote summary:
  - activation ratio `MLXs/upstream ~0.93147x`
  - first-step ratio `~2.16977x`
  - total ratio `~0.95717x`
- Outcome:
  - prompt/handoff ownership is not the first bounded slice
  - next active slice becomes `R24-S2 current fast-path first-step decomposition`

## 2026-04-20 — M01.R24-S2 active-batch first-step candidate rejected

- Candidate implemented in the `R24` worktree only:
  - removed grouped `mx.eval(prepared.tokens, next_prepared.tokens)` from `materialize_prepared_batch_step()`
  - left scheduling and row append unchanged
- Focused correctness:
  - `5 passed` on `tests/unit/test_runtime_core/test_batch_progression.py`
  - `ruff check` passed on touched runtime/tests/probes
- Local micro evidence:
  - first-step control: `/tmp/mlxs_m01_r24_fast_batch_first_step_local_20260420_1.json`
  - first-step candidate: `/tmp/mlxs_m01_r24_fast_batch_first_step_candidate_local_20260420_1.json`
  - materialization boundary improved strongly on the micro probe
- Remote micro evidence:
  - first-step control: `/tmp/mlxs_m01_r24_fast_batch_first_step_remote_20260420_1.json`
  - first-step candidate: `/tmp/mlxs_m01_r24_fast_batch_first_step_candidate_remote_20260420_2.json`
  - result:
    - total `candidate/control ~0.45550x`
    - `materialize_s candidate/control ~0.14887x`
- Authoritative remote scheduler guardrail:
  - control: `/tmp/mlxs_m01_r24_control_remote_scheduler_20260420_1.json`
  - candidate: `/tmp/mlxs_m01_r24_candidate_remote_scheduler_20260420_1.json`
  - prompt `256 candidate/control requests_per_s ~0.95880x`
  - prompt `2048 candidate/control requests_per_s ~1.04748x`
- Classification:
  - non-promotable
  - the one-step materialization boundary was not an independent lever; it was carrying next-step overlap/readiness
- Outcome:
  - candidate reverted cleanly in the worktree
  - next active slice becomes `R24-S3 active-batch overlap decomposition`

## 2026-04-20 — M01.R24-S3 active-batch overlap decomposition started

- New local artifact:
  - `/tmp/mlxs_m01_r24_fast_batch_overlap_local_20260420_1.json`
- Initial local readback:
  - MLXs step 0 total `~0.08893s` vs upstream step 0 core `~0.06332s`
  - MLXs step 1 total `~0.02028s` vs upstream step 1 core `~0.02408s`
  - step 0 still pays the materialization tax
  - step 1 gains from readiness / overlap
- Interpretation:
  - `R24-S2` failed because the one-step materialization cut removed useful overlap
  - next bounded work should target overlap/readiness ownership directly

## 2026-04-20 — M01.R24-S3 classified and advanced

- Remote overlap artifact:
  - `/tmp/mlxs_m01_r24_fast_batch_overlap_remote_20260420_1.json`
- Remote readback:
  - step 0 total `MLXs/upstream ~2.16030x`
  - step 1 total `MLXs/upstream ~1.03131x`
  - step 1 scheduling/core alone `MLXs/upstream ~0.13768x`
- Classification:
  - not a narrow overlap fix
  - broader active-batch state-lifetime redesign
- Outcome:
  - `R24-S3` is complete as a decisive decomposition step
  - next active path initially became `R24-S4 active-batch state-lifetime redesign`

## 2026-04-20 — state repair after `R24-S4` inconsistency

- Problem:
  - `R24-S4` was left active in the file-backed system
  - but its first concrete candidate had already been attempted, failed local truth-first, and been reverted
- Rejected candidate now made explicit:
  - `R24-S4a resident dual-step _SharedFastBatch owner`
- Decision:
  - Outcome B
  - keep the broader batch family alive
  - open a genuinely new path with a new identifier and a different thesis
- New active path:
  - `R25 canonical prompt-batch + extendable active-batch ownership redesign`

## 2026-04-20 — M01.R25 owner-boundary redesign rejected locally and reranked

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r25-canonical-batch-ownership`
- Branch:
  - `codex/m01-r25-canonical-batch-ownership`
- Baseline:
  - commit `fcb69de`
- Candidate family attempted:
  - per-row-offset batch-cache substrate for the fast path
  - token-resident extendable active owner
  - dual-ready extendable active owner
- Focused correctness before rejection:
  - `12 passed` on `tests/unit/test_cache/test_batch_kv.py tests/unit/test_runtime_core/test_batch_progression.py tests/unit/test_batch/test_scheduler.py`
  - `ruff check` passed on the touched runtime/tests files
- Truth-first local scheduler result on prompt `256`:
  - control artifact: `/tmp/mlxs_m01_r25_scheduler_control_local_20260420_1.json`
  - first candidate artifact: `/tmp/mlxs_m01_r25_scheduler_candidate_local_20260420_1.json`
  - second candidate artifact: `/tmp/mlxs_m01_r25_scheduler_candidate_local_20260420_2.json`
  - first candidate requests/s `candidate/control ~0.79339x`
  - second candidate requests/s `candidate/control ~0.73350x`
- Additional benchmark tooling created in the isolated worktree:
  - `benchmarks/mlxs_vs_mlx_lm/r25_staggered_batch_compare.py`
- New staggered local compare:
  - artifact: `/tmp/mlxs_m01_r25_staggered_local_20260420_1.json`
  - prompt `256 mlxs/mlx_lm requests_per_s ~0.69428x`
  - exact output parity `True`
- Outcome:
  - reject the `R25` owner family on the current canonical static surface
  - rerank to `R26 dual-owner batch architecture rerank on staggered extension surface`

## 2026-04-20 — M01.R26 remote authority and decomposition established

- Remote accepted tree check failed the playbook assumptions:
  - `/tmp/mlxs_layer1_candidate_20260417_1` was on the wrong commit and dirty
- Response:
  - created fresh isolated remote baseline:
    - `/tmp/mlxs_m01_r26_remote_baseline_20260420_1`
  - synced from local `fcb69de` worktree plus bounded `R26` benchmark tooling only
- Remote staggered compare:
  - artifact: `/tmp/mlxs_m01_r26_staggered_remote_20260420_1.json`
  - result:
    - `mlxs/mlx_lm requests_per_s ~0.85088x`
    - exact output parity `True`
- Remote staggered decomposition:
  - artifact: `/tmp/mlxs_m01_r26_staggered_decomp_remote_20260420_1.json`
  - decisive readback:
    - MLXs late request first-token step `129`
    - `mlx_lm` late request first-token step `4`
    - MLXs late request completion step `255`
    - `mlx_lm` late request completion step `131`

## 2026-04-20 — M01.R26 runtime slices rejected and reranked

- `R26-S1` full dynamic late-admission owner:
  - local control artifact: `/tmp/mlxs_m01_r26_staggered_control_local_20260420_1.json`
  - local candidate artifact: `/tmp/mlxs_m01_r26_staggered_candidate_local_20260420_1.json`
  - result:
    - requests/s `candidate/control ~0.77448x`
    - exact output parity `False`
  - classification:
    - rejected locally, no remote budget
- `R26-S2` concurrent late prefill plus legacy continuation handoff:
  - local control artifact: `/tmp/mlxs_m01_r26b_staggered_control_local_20260420_1.json`
  - local candidate artifact: `/tmp/mlxs_m01_r26b_staggered_candidate_local_20260420_1.json`
  - result:
    - requests/s `candidate/control ~0.99397x`
    - p50 TTFT `~0.94485x`
    - p95 completion `~1.04957x`
    - exact output parity `True`
  - local candidate decomposition:
    - `/tmp/mlxs_m01_r26b_staggered_decomp_candidate_local_20260420_1.json`
    - late request first-token step `129 -> 2`
    - late request completion step `255 -> 128`
  - classification:
    - rejected as neutral-to-regressive
- Outcome:
  - prompt-side admission semantics are correct
  - synchronous full prompt work inside the main scheduler step is still too expensive
  - next active path becomes `R26-S4 integrated prompt-owner budgeted progression redesign`

## 2026-04-20 — M01.R26-S4 budgeted prompt-owner redesign rejected locally

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r26-dual-owner-batch`
- Candidate shape:
  - real prompt-side owner
  - chunked/budgeted late prompt progress
  - dedicated MLX stream for prompt progress
  - `_SharedFastBatch` preserved and hot
- Focused correctness:
  - `9 passed` on `tests/unit/test_runtime_core/test_batch_progression.py tests/unit/test_batch/test_scheduler.py`
  - `ruff check` passed on touched runtime/tests/benchmark files
- Local staggered control/candidate:
  - control: `/tmp/mlxs_m01_r26s4_staggered_control_local_20260420_1.json`
  - candidate: `/tmp/mlxs_m01_r26s4_staggered_candidate_local_20260420_1.json`
  - requests/s `candidate/control ~0.91001x`
  - p50 TTFT `~0.99142x`
  - p95 completion `~1.14951x`
  - exact output parity `True`
- Candidate decomposition:
  - `/tmp/mlxs_m01_r26s4b_staggered_decomp_candidate_local_20260420_1.json`
  - late first-token step `4`
  - late completion step `130`
- Outcome:
  - reject `R26-S4`
  - rerank to `R26-S5 true-overlap prompt-progress substrate redesign`

## 2026-04-20 — M01.R26-S5 lower-layer prompt-progress redesign rejected and family closed

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r26-dual-owner-batch`
- Candidate shape:
  - lower-layer prompt progression object in `runtime_core`
  - explicit Layer 1 stream ownership for the hot fast batch and prompt progression
  - preserved prompt-owner admission semantics
- Focused correctness:
  - `9 passed` on `tests/unit/test_runtime_core/test_batch_progression.py tests/unit/test_batch/test_scheduler.py`
  - `ruff check` passed on touched runtime/tests/benchmark files
- Local staggered control/candidate:
  - control: `/tmp/mlxs_m01_r26s5_staggered_control_local_20260420_1.json`
  - candidate: `/tmp/mlxs_m01_r26s5_staggered_candidate_local_20260420_1.json`
  - requests/s `candidate/control ~1.00040x`
  - p50 TTFT `~1.04423x`
  - p95 completion `~1.04108x`
  - exact output parity `True`
- Candidate decomposition:
  - `/tmp/mlxs_m01_r26s5_staggered_decomp_candidate_local_20260420_1.json`
  - late first-token step `4`
  - late completion step `130`
- Outcome:
  - reject `R26-S5`
  - close the `R26` batch prompt-progress family
  - rerank globally to `R27`

## 2026-04-20 — M01.R27 closure check complete, R28 opened

- Global rerank conclusion:
  - current front `R26` was sufficiently understood
  - it had already exhausted more than 2-3 bounded families
  - closure/rerank was correct
- Strongest remaining open front:
  - `AC1`
- Next-move decision:
  - choose integrated refactor, not another bounded helper/core slice
- New isolated worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r28-ac1-integrated-core-refactor`
- Branch:
  - `codex/m01-r28-ac1-integrated-core-refactor`
- Baseline:
  - commit `fcb69de`

## 2026-04-20 — M01.R28 integrated AC1 contract refactor rejected locally

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r28-ac1-integrated-core-refactor`
- Candidate shape:
  - canonical single-request progression contract in Layer 1
  - benchmark Class A and `runtime_core.run_greedy` consume the same progression loop
  - benchmark-specific lookahead policy moved to the contract boundary
- Focused correctness:
  - `16 passed` on `tests/unit/test_runtime_core/test_decode.py tests/unit/test_runtime_core/test_greedy.py tests/unit/test_benchmarks/test_backends.py`
  - `ruff check` passed on the touched runtime/benchmark files
- Local accepted-sensitive package:
  - control: `/tmp/mlxs_m01_r28_control_local_20260420_2.json`
  - candidate: `/tmp/mlxs_m01_r28_candidate_local_20260420_2.json`
  - `Llama-3.2-1B 256 decode ~0.98852x`
  - `Llama-3.2-1B 2048 decode ~0.99909x`
  - `Qwen2.5-1.5B 256 decode ~0.90711x`
  - `Qwen2.5-1.5B 2048 decode ~0.92558x`
- Outcome:
  - reject `R28`
  - rerank to `R29 AC1 lower-level substrate and capability rerank`

## 2026-04-20 — M01.R29 AC1 substrate rerank closes AC1 for this phase

- Main artifact:
  - `/tmp/mlxs_m01_r29_substrate_local_20260420_1.json`
- Probe shape:
  - no-patch accepted-sensitive local AC1 substrate probe
  - exact parity/count/prefix checks
  - lower-boundary timing on:
    - model forward
    - attention
    - cache update/fetch
    - mask creation
    - eval / async-eval
    - selection / materialization
- Result:
  - parity exact on all local cases
  - overall decode stayed near-parity / mixed rather than exposing a strong remaining AC1 lever
  - component deltas were mixed enough that neither a built-in MLX path nor a custom-kernel path is justified for AC1
- Outcome:
  - close `AC1` for this phase
  - rerank globally to `R30 AC2 lower-level substrate and capability rerank on the staggered batch surface`

## 2026-04-20 — M01.R30 opened

- New strongest active front:
  - `AC2` on the remote-authoritative staggered late-admission surface
- Next move:
  - rerank below the closed prompt-owner / prompt-progress family
  - classify whether the remaining batch tax is internal lower-layer, built-in MLX capability, custom extension / Metal kernel, or exhausted

## 2026-04-20 — M01.PC1 repo-level process correction applied

- Scope:
  - stop runtime/performance implementation in this turn
  - repair repo-level operating policy so future MLXs work cannot drift back into micro-family churn
- Result:
  - strengthened `AGENTS.md`
  - strengthened `docs/execution/REDESIGN_MODE.md`
  - added binding `docs/execution/REFACTOR_POLICY.md`
  - updated file-backed execution docs to record that the old process was too slice-fragmented
- Consequence:
  - future runtime/performance work must use:
    - small-step work only for bottleneck confirmation
    - integrated refactor or closure/rerank once a front is sufficiently understood and has burned about `2-3` bounded families without a strong win
    - one main runtime/refactor front at a time
  - next technical front remains `R30`

## 2026-04-20 — worktree / branch hygiene normalized after `M01.PC1`

- Pruned stale worktree record:
  - `/private/tmp/mlxs_longcase_control`
- Removed clean closed M01 worktrees and deleted their branches:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r17-ac1-short-prompt-alignment`
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r18-ac1-shared-short-prompt`
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r19-ac1-inline-short-prompt`
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r20-ac1-helper-hotloop`
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r21-ac1-schedule-next-contract`
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r28-ac1-control`
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r28-ac1-integrated-core-refactor`
- Retained active baseline worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-core-exec`
  - branch `refactor/core-exec`
  - reason: file-backed execution truth and neutral probe/tooling changes remain dirty here
- Retained active technical worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r30-ac2-substrate-rerank`
  - branch `codex/m01-r30-ac2-substrate-rerank`
  - reason: this is the one dedicated runtime worktree for the current `R30` front
- Retained dirty historical M01 worktrees:
  - `R22`, `R23`, `R24`, `R25`, `R26`, `R29`
  - reason: each still holds unreviewed local probe/docs state and is not safe for autonomous deletion yet

## 2026-04-20 — M01.R30 lower-boundary probe completed and reranked

- Added repo-native probe:
  - `benchmarks/mlxs_vs_mlx_lm/r30_ac2_substrate_probe.py`
  - `tests/unit/test_benchmarks/test_r30_ac2_substrate_probe.py`
- Validation:
  - `uv run ruff check benchmarks/mlxs_vs_mlx_lm/r30_ac2_substrate_probe.py tests/unit/test_benchmarks/test_r30_ac2_substrate_probe.py`
  - `.venv/bin/python -m pytest tests/unit/test_benchmarks/test_r30_ac2_substrate_probe.py`
- Local probe artifacts:
  - `/tmp/mlxs_m01_r30_substrate_local_20260420_1.json`
  - `/tmp/mlxs_m01_r30_substrate_local_20260420_2.json`
- Stable readback across both seeds:
  - exact output parity: `True`
  - MLXs late request first-token/completion steps: `129/255`
  - `mlx_lm` late request first-token/completion steps: `4/131`
  - MLXs fast-batch step remained dominated by `mx.eval` (`~0.892s`) with much smaller `mx.async_eval` (`~0.144-0.152s`)
  - upstream generation step remained much heavier on `mx.async_eval` (`~0.919-0.925s`) with much smaller `mx.eval` (`~0.346s`)
  - attention, mask, and cache-update timings stayed tiny on both sides relative to the evaluation-boundary split
- Outcome:
  - `R30` does not justify a standalone built-in-MLX or custom-kernel pivot
  - the exact next path is lower-level internal batch redesign centered on generation-owner / eval-discipline ownership below the closed prompt-owner family

## 2026-04-20 — M01.R30 first integrated runtime candidate rejected and reranked

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r30-ac2-substrate-rerank`
  - branch `codex/m01-r30-ac2-substrate-rerank`
- Candidate shape:
  - resident current/next token lifetime in the fast generation owner
  - prompt-prefill handoff into a new fast generation owner
  - scheduler orchestration over multiple lower-level fast owners
- Focused local validation in the dedicated worktree:
  - `uv run ruff check src/mlxs/runtime_core/batch_progression.py src/mlxs/batch/fast_batch.py src/mlxs/batch/scheduler.py tests/unit/test_runtime_core/test_batch_progression.py tests/unit/test_batch/test_scheduler.py`
  - `PYTHONPATH=src uv run pytest tests/unit/test_runtime_core/test_batch_progression.py tests/unit/test_batch/test_scheduler.py tests/unit/test_product_surfaces/test_batched_serving.py`
- Local staggered truth-first compare:
  - `/tmp/mlxs_m01_r30_staggered_local_20260420_2.json`
  - result:
    - requests/s `~0.70730x`
    - generated_tok/s `~0.70730x`
    - p50 TTFT `~0.94736x`
    - p95 completion `~1.41507x`
    - exact parity `True`
- Local staggered decomposition:
  - `/tmp/mlxs_m01_r30_staggered_decomp_local_20260420_2.json`
  - result:
    - MLXs late first-token/completion steps now `4/131`
    - `mlx_lm` late first-token/completion steps `4/131`
- Candidate lower-boundary probe:
  - `/tmp/mlxs_m01_r30_substrate_candidate_local_20260420_1.json`
  - decisive readback:
    - MLXs `fast_batch_step` calls `256`
    - upstream `generation_batch_step` calls `133`
    - semantics were repaired by parallel generation owners, but active generation work was duplicated
- Outcome:
  - candidate rejected locally
  - dedicated worktree reverted cleanly
  - next exact path reranked to one active extendable dynamic generation owner on the staggered surface

## 2026-04-21 — M01.R30 last acceptable candidate rejected and family closed

- Worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r30-ac2-substrate-rerank`
  - branch `codex/m01-r30-ac2-substrate-rerank`
- Candidate shape:
  - one active extendable dynamic generation owner
  - dynamic-only cache substrate
  - no parallel fast generation owners
- Focused local validation in the dedicated worktree:
  - `uv run ruff check src/mlxs/cache/kv.py src/mlxs/cache/__init__.py src/mlxs/runtime_core/batch_progression.py src/mlxs/batch/fast_batch.py src/mlxs/batch/scheduler.py tests/unit/test_cache/test_kv.py tests/unit/test_batch/test_scheduler.py`
  - `PYTHONPATH=src uv run pytest tests/unit/test_cache/test_kv.py tests/unit/test_runtime_core/test_batch_progression.py tests/unit/test_batch/test_scheduler.py tests/unit/test_product_surfaces/test_batched_serving.py`
- Local staggered truth-first compare:
  - `/tmp/mlxs_m01_r30_staggered_local_20260421_1.json`
  - result:
    - requests/s `~0.72761x`
    - generated_tok/s `~0.72761x`
    - p50 TTFT `~0.97741x`
    - p95 completion `~1.37504x`
    - exact parity `True`
- Local staggered decomposition:
  - `/tmp/mlxs_m01_r30_staggered_decomp_local_20260421_1.json`
  - result:
    - MLXs late first-token/completion steps `4/131`
    - `mlx_lm` late first-token/completion steps `4/131`
- Candidate lower-boundary probe:
  - `/tmp/mlxs_m01_r30_substrate_candidate_local_20260421_1.json`
  - result:
    - one-owner semantics hold
    - lower-boundary wall-time on the mixed-offset dynamic path remains too high
- Outcome:
  - candidate rejected locally
  - `R30` family closed for this phase
  - dedicated worktree reverted cleanly, removed, and branch deleted

## 2026-04-21 — post-`R30` rerank and new worktree

- New active broader path:
  - `M01.R31 mixed-offset batch cache substrate redesign on the staggered surface`
- Fresh dedicated worktree opened:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r31-mixed-offset-cache-substrate`
  - branch `codex/m01-r31-mixed-offset-cache-substrate`
  - base commit `fcb69de`
