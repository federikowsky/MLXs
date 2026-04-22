# Decisions

Append-only. Newest entries go last.

## D001 — 2026-04-19 — accepted

- Decision: `docs/execution/` becomes the file-backed system of record.
- Why: important project state must not live only in chat.
- Evidence: user directive; accumulated cross-family state is too large and long-lived for chat-only continuity.

## D002 — 2026-04-19 — accepted

- Decision: authoritative promotable performance claims remain remote-only.
- Why: local probes are useful for isolation; accepted claims must run on the approved remote benchmark host.
- Evidence: [docs/refactor/benchmark-protocol.md](../refactor/benchmark-protocol.md), accepted remote artifact history.

## D003 — 2026-04-19 — accepted

- Decision: compile decode eligibility stays prompt-aware at `<=512`.
- Why: longer prompts regressed under compiled decode; short prompts were stable.
- Evidence: promoted compile stabilization artifacts and tests.

## D004 — 2026-04-19 — accepted

- Decision: Qwen flat-config compatibility is fixed in `src/mlxs/models/qwen.py`, not in a RoPE workaround.
- Why: the first inconsistency was metadata reconstruction in `resolved_text_args()`, not the RoPE kernel.
- Evidence: Qwen compatibility investigation and accepted remote reruns.

## D005 — 2026-04-19 — accepted

- Decision: Qwen long-prompt decode improvement stays benchmark-surface scoped.
- Why: generic long prepared-step regressed accepted Llama guardrails; Qwen-scoped helper was positive and bounded.
- Evidence: `/tmp/mlxs_qwen_long_prepared_llama_guardrail_local_20260418_1.json`, promoted Qwen remote artifacts.

## D006 — 2026-04-19 — accepted

- Decision: Llama-3.2-3B long-prompt decode improvement stays benchmark-surface scoped.
- Why: 3B-scoped helper improved remote 3B slices while keeping Llama 1B and Qwen no-op guardrails effectively neutral.
- Evidence: `/tmp/mlxs_ac1_llama32_3b_long_prepared_probe_remote_20260418_1.json`, `/tmp/mlxs_ac1_llama3b_long_guardrails_confirm_20260418_1.json`, `/tmp/mlxs_class_a_llama32_3b_promoted_remote_20260418_1.json`.

## D007 — 2026-04-19 — accepted

- Decision: AC2 batch-host restoration is structural cleanup, not a throughput lever.
- Why: restored host is runnable and near-neutral vs scheduler baseline.
- Evidence: `/tmp/mlxs_ac2_batch_host_final_compare_20260418_1.json`.

## D008 — 2026-04-19 — accepted

- Decision: AC13 is blocked-for-now on the accepted baseline.
- Why: fair prefill-step tuning stayed below target and bounded prefill candidates were effectively no-ops.
- Evidence: `/tmp/mlxs_ac13_prefillstep_ranked_20260418_1.json`, `/tmp/mlxs_ac13_chunkcap_probe_local_20260418_2.json`, `/tmp/mlxs_ac13_streamhoist_probe_local_20260418_1.json`.

## D009 — 2026-04-19 — accepted

- Decision: AC1 remains open because of true performance gaps on the current broadened accepted model set, not because reference breadth was too narrow.
- Why: adding `mlx-community/Llama-3.2-3B-Instruct-4bit` still left all accepted references below `1.10x decode`.
- Evidence: `/tmp/mlxs_class_a_accepted_remote_20260418_1.json`, `/tmp/mlxs_class_a_qwen15b_promoted_remote_20260418_3.json`, `/tmp/mlxs_class_a_llama32_3b_promoted_remote_20260418_1.json`.

## D010 — 2026-04-19 — accepted

- Decision: start `M01-performance-architecture-redesign`.
- Why: local bounded seam families are mostly exhausted; acceptance gaps now point to structural hot-path issues.
- Evidence: AC1/AC2 remain open after accepted seam wins; AC13 blocked; scheduler micro-families and long-decode micro-families mostly exhausted or scoped to benchmark helper only.

## D011 — 2026-04-19 — accepted

- Decision: the first M01 redesign front is `resident decode state machine / hot-path unification`.
- Why: benchmark helper, `runtime_core`, `general_path`, and `batch.scheduler` currently own divergent decode progression loops; accepted benchmark wins live above Layer 1 instead of inside a shared resident progression contract.
- Evidence: `src/mlxs/runtime_core/greedy.py`, `src/mlxs/runtime_core/decode.py`, `src/mlxs/general_path/single_request.py`, `src/mlxs/batch/scheduler.py`, `benchmarks/mlxs_vs_mlx_lm/backends.py`.

## D012 — 2026-04-19 — accepted

- Decision: the target redesign boundary is a Layer 1 resident decode progression object, not a richer event contract.
- Why: the core must own cache/state/execution/next-step progression, while text decode, stop-sequence text matching, logprob shaping, and `TokenEvent` construction must stay above Layer 1.
- Evidence: `docs/refactor/performance-core-spec.md`, `src/mlxs/runtime_core/contracts.py`, `src/mlxs/_types.py`, `src/mlxs/general_path/single_request.py`, `src/mlxs/batch/scheduler.py`.

## D013 — 2026-04-19 — rejected

- Decision: reject the first implementation shape `class-based DecodeProgression.step()` as the initial resident decode slice.
- Why: focused local correctness held, but the isolated remote candidate regressed the accepted lookahead-sensitive benchmark slices sharply.
- Evidence: `/tmp/mlxs_m01_resident_decode_candidate_remote_20260419_1.json`, `/tmp/mlxs_m01_resident_decode_candidate_compare_20260419_1.json`.

## D014 — 2026-04-19 — accepted

- Decision: the next redesign slice should keep the resident progression goal but use a lower-overhead shape: data-only progression state with module-level advance helpers, migrating the benchmark helper first only.
- Why: benchmark helper remains the highest-leverage first consumer, but the class-based hot-loop shape is now proven too expensive.
- Evidence: D013 rejection, accepted benchmark-helper wins already living above Layer 1, and the need to avoid broad multi-consumer migration on the next slice.

## D015 — 2026-04-19 — accepted

- Decision: the next slice will target a minimal data-only Layer 1 progression state carrying only `CoreState`, `CoreTerminationPolicy`, `CoreExecutionPolicy`, `StepFn`, `TokenSelector`, `use_lookahead`, and the resident current-step handle (`logits` or prepared token/logits).
- Why: this is the smallest contract that can move decode progression ownership into Layer 1 while avoiding the rejected class/method hot-loop shape.
- Evidence: D012 boundary, D013 rejection, live reads of `runtime_core.decode`, benchmark helper, and `general_path`.

## D016 — 2026-04-19 — rejected

- Decision: reject the low-overhead data-only progression state as the first migrated slice.
- Why: after fixing the local benchmark integration bug, accepted-sensitive local guardrails still regressed materially on Llama 1B `256` and Qwen `256/2048`.
- Evidence: `/tmp/mlxs_m01_low_overhead_local_guardrails_20260419_2.json`, compared against `/tmp/mlxs_m01_low_overhead_main_guardrails_20260419_1.json`.

## D017 — 2026-04-19 — accepted

- Decision: the next adjacent redesign slice is `inline resolved-step progression helper migration`.
- Why: both container-based shapes regressed; the remaining bounded path is to keep the hot loop inline while reusing only the resolved-step helper primitives now extracted in `runtime_core.decode`.
- Evidence: D013, D016, retained local-neutral groundwork in `src/mlxs/runtime_core/decode.py`.

## D018 — 2026-04-19 — rejected

- Decision: reject `M01.4 inline resolved-step progression helper migration` as a promotable slice.
- Why: local guardrails were mixed; Llama improved, but Qwen `2048` still regressed on the accepted-sensitive confirm.
- Evidence: `/tmp/mlxs_m01_inline_helpers_local_guardrails_20260419_1.json`, `/tmp/mlxs_m01_inline_helpers_worktree_qwen_confirm_20260419_1.json`, `/tmp/mlxs_m01_inline_helpers_main_qwen_confirm_20260419_1.json`.

## D019 — 2026-04-19 — accepted

- Decision: the next adjacent redesign slice is `lookahead-branch-only helper migration ranking`.
- Why: container-based slices failed broadly, while the inline helper slice was mixed with the remaining loss concentrated on the Qwen long lookahead path; the next useful step is to isolate the lookahead branch itself, not both decode branches together.
- Evidence: D018 rejection plus the retained neutral helper extraction in `src/mlxs/runtime_core/decode.py`.

## D020 — 2026-04-19 — accepted

- Decision: promote `M01.5 lookahead-branch-only helper migration`.
- Why: the accepted-sensitive interleaved local compare was neutral-to-better on all four cases, and the accepted-remote rerun stayed neutral-to-slightly-better on all four accepted-sensitive slices.
- Evidence: `/tmp/mlxs_m01_lookahead_branch_probe_local_20260419_1.json`, `/tmp/mlxs_m01_lookahead_branch_interleaved_compare_local_20260419_1.json`, `/tmp/mlxs_m01_lookahead_branch_promoted_remote_20260419_1.json`, `/tmp/mlxs_m01_lookahead_branch_promoted_compare_20260419_1.json`.

## D021 — 2026-04-19 — accepted

- Decision: the next adjacent redesign path is `runtime_core.run_greedy consumer ranking`.
- Why: benchmark helper is now on the retained lookahead helper groundwork, so the next highest-leverage and still-bounded consumer is the canonical Layer 1 entrypoint.
- Evidence: `src/mlxs/runtime_core/greedy.py`, promoted `M01.5` state, and the need to move accepted benchmark wins closer to the real core.

## D022 — 2026-04-19 — accepted

- Decision: pause further implementation slicing and switch the program to `M01.R1 deep architecture research`.
- Why: the project has enough negative evidence from bounded seam families that more local candidate churn is lower-value than a full architecture reconstruction.
- Evidence: frozen rejected families in `docs/execution/FREEZE.md`, open AC1/AC2, blocked AC13, and the user program directive.

## D023 — 2026-04-19 — accepted

- Decision: no new performance implementation slice is allowed until `M01.R1` is complete and recorded in the file-backed system.
- Why: redesign choices now need a full MLXs vs `mlx_lm` vs MLX substrate picture rather than another local tweak.
- Evidence: user program directive and current negative evidence set.

## D024 — 2026-04-19 — accepted

- Decision: the ranked redesign thesis is `core-first resident progression unification`.
- Why: the clearest structural tax is the benchmark/core split: accepted fast-path wins live in the benchmark helper while `runtime_core.run_greedy` still uses the simpler baseline loop; this split also blocks a cleaner batch redesign later.
- Evidence: `docs/execution/milestones/M01-performance-architecture-redesign/RESEARCH.md`, `docs/execution/milestones/M01-performance-architecture-redesign/ARCHITECTURE_NOTES.md`.

## D025 — 2026-04-19 — accepted

- Decision: the fallback redesign thesis is `batch-first resident progression redesign`.
- Why: AC2 still has the largest open throughput gap, and `mlx_lm.BatchGenerator` has a clearly more resident batch architecture than MLXs `BatchScheduler`; this becomes the next fallback only if the core-first progression unification does not yield enough leverage.
- Evidence: `RESEARCH.md` Phase 5, `ARCHITECTURE_NOTES.md` batch maps, `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`, `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`.

## D026 — 2026-04-19 — accepted

- Decision: the first bounded implementation slice after research is `runtime_core.run_greedy lookahead-path adoption`.
- Why: it moves an already-promoted benchmark-helper win into the real Layer 1 consumer with the narrowest blast radius, without reopening rejected container shapes or touching `general_path`/`batch`.
- Evidence: `RESEARCH.md` final section, `ARCHITECTURE_NOTES.md` MLXs single-request maps, promoted `M01.5` artifacts.

## D027 — 2026-04-19 — accepted

- Decision: the `M01.R1b MLX substrate capability inventory` addendum does not change the ranked redesign thesis.
- Why: the wider MLX capability surface adds useful later options, but no capability family displaces the current top thesis of core-first resident progression unification.
- Evidence: `docs/execution/milestones/M01-performance-architecture-redesign/MLX_CAPABILITY_INVENTORY.md`, `RESEARCH.md` addendum section, official MLX docs and installed runtime inventory.

## D028 — 2026-04-19 — accepted

- Decision: promote `M01.R2 runtime_core.run_greedy lookahead-path adoption`.
- Why: the bounded `run_greedy` slice was neutral-to-better on the interleaved local compare and neutral-to-better on the authoritative remote compare across the accepted-sensitive Llama 1B and Qwen 1.5B slices.
- Evidence: `/tmp/mlxs_m01_r2_run_greedy_interleaved_compare_local_20260419_1.json`, `/tmp/mlxs_m01_r2_run_greedy_candidate_compare_remote_20260419_1.json`.

## D029 — 2026-04-19 — accepted

- Decision: rank `general_path single_request` ahead of `batch.scheduler` as the next redesign consumer after `run_greedy`.
- Why: `general_path` already consumes Layer 1 state and helpers, duplicates prepared-step progression directly, has materially richer direct tests, and carries much lower blast radius than `BatchScheduler`, which still owns batch-state rebuild and `TokenEvent` shaping in the scheduler loop.
- Evidence: `src/mlxs/general_path/single_request.py`, `src/mlxs/batch/scheduler.py`, `tests/unit/test_general_path/test_single_request.py`, `tests/unit/test_batch/test_scheduler.py`, `tests/unit/test_product_surfaces/test_batched_serving.py`.

## D030 — 2026-04-19 — accepted

- Decision: the first bounded `general_path single_request` slice is `prepared-step branch helper adoption`.
- Why: the smallest safe move is to migrate only the two branches that already schedule the next token (`use_prepared_step_enriched_path` and `use_sampled_prepared_step_path`) to the retained resolved-step helper, while leaving the processor path and baseline decode path untouched.
- Evidence: `src/mlxs/general_path/single_request.py`, `tests/unit/test_general_path/test_single_request.py`, promoted `M01.5` and `M01.R2` helper-groundwork wins.

## D031 — 2026-04-19 — rejected

- Decision: reject the broad `general_path prepared-step branch helper adoption` slice as a promotable slice.
- Why: the broad slice was locally mixed; Llama short logprobs and Qwen long top-logprobs improved, sampled stayed neutral, but Qwen short plain logprobs regressed on focused confirm.
- Evidence: `/tmp/mlxs_m01_r3_general_path_interleaved_compare_local_20260419_1.json`, `/tmp/mlxs_m01_r3_qwen_short_logprobs_confirm_20260419_1.json`.

## D032 — 2026-04-19 — accepted

- Decision: the next adjacent redesign path is `general_path enriched top-logprobs-only helper migration ranking`.
- Why: the positive signal concentrated in the enriched top-logprobs subpath, while the broader enriched+sampled slice was too wide and let the Qwen short plain-logprobs case regress.
- Evidence: D031 rejection and the branch-local result split across `general_path` cases.

## D033 — 2026-04-19 — rejected

- Decision: reject the narrowed `general_path enriched top-logprobs-only helper migration` slice as a promotable slice.
- Why: the narrowed slice was still too weak/mixed: short and long `top_logprobs` cases stayed near-neutral or slightly negative, with no strong enough signal to justify remote budget.
- Evidence: `/tmp/mlxs_m01_r4_toplogprobs_interleaved_compare_local_20260419_1.json`.

## D034 — 2026-04-19 — accepted

- Decision: `general_path` follow-up slices are exhausted for this phase; the next adjacent redesign path should move to `resident batch progression / AC2 throughput architecture`.
- Why: both the broad `general_path` slice and the narrower `top_logprobs` slice failed to produce a clean strong enough signal, so further narrowing inside `general_path` is lower-leverage than moving to the remaining structural AC2 path.
- Evidence: D031, D033, `RESEARCH.md` ranked thesis and fallback, `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`, `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`.

## D035 — 2026-04-19 — accepted

- Decision: the first batch-side redesign candidate space should be ranked around `resident active-batch progression` rather than another scheduler micro-fix or product-surface host tweak.
- Why: accepted AC2 artifacts and side-by-side code read show the dominant remaining batch tax is structural ownership of active batch state and progression, not merge/scatter alone and not the restored batch host surface.
- Evidence: `src/mlxs/batch/scheduler.py`, installed `mlx_lm/generate.py::BatchGenerator`, `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`, `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`, `/tmp/mlxs_ac2_batch_host_final_compare_20260418_1.json`.

## D036 — 2026-04-19 — accepted

- Decision: rank batch architecture candidates as:
  1. `resident active completion-batch object`
  2. `prefill/prompt-processing unification around a resident batch object`
- Why: AC2 short-prompt cost is decode-dominated and AC2 long-prompt cost still has a large decode share, so the first bounded slice should attack resident completion ownership before widening into prompt-processing redesign.
- Evidence: `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`, `src/mlxs/batch/scheduler.py`, installed `mlx_lm/generate.py::BatchGenerator`.

## D037 — 2026-04-19 — accepted

- Decision: the first bounded batch-side implementation slice after ranking is `resident active completion-batch object slice definition`.
- Why: it is the smallest architecture slice that can test the resident-batch thesis without rewriting prefill, queue admission, or product-surface host integration all at once.
- Evidence: D035, D036, `ARCHITECTURE_NOTES.md` batch map.

## D038 — 2026-04-19 — accepted

- Decision: the resident active completion-batch object should carry only:
  - ordered `entries` / request-sequence references
  - merged resident `cache`
  - current batched token ids `y`
  - optional resident `logprobs` only if the simple-path implementation needs them for sampling/future step continuity
  - resident generation counts / max-token counters only if not derivable cheaply from `_Sequence`
- Why: this is the smallest state that preserves completion-step ownership without swallowing prompt-processing, queue admission, product event history, or prompt-cache policy.
- Evidence: `src/mlxs/batch/scheduler.py`, installed `mlx_lm/generate.py::Batch`, `RESEARCH.md`, `ARCHITECTURE_NOTES.md`.

## D039 — 2026-04-19 — accepted

- Decision: the first bounded batch-side slice excludes:
  - prompt processing / prompt checkpointing
  - prompt-cache prepare/finalize integration
  - product-surface stream/result plumbing
  - batch-wide event/logprob API redesign
  - scheduler admission/rejection policy changes
- Why: those would broaden the slice beyond the first resident completion-batch experiment.
- Evidence: D037 and the current `BatchScheduler` / `BatchServingHost` ownership map.

## D040 — 2026-04-19 — rejected

- Decision: reject the no-patch `resident active completion-batch object` slice as the first batch-side implementation path.
- Why: the truth-first probe on the canonical AC2 scheduler surface was regressive on both prompt sizes and failed parity.
- Evidence: `/tmp/mlxs_m01_r6_completion_batch_probe_local_20260419_1.json`.

## D041 — 2026-04-19 — accepted

- Decision: move to the fallback batch-side path `prefill/prompt-processing unification around a resident batch object`.
- Why: the first-ranked completion-batch slice did not clear the truth-first gate; the next adjacent structural path is the ranked fallback from D036.

## D042 — 2026-04-19 — accepted

- Decision: the first bounded `R7` target is `resident prefill-to-completion handoff for aligned cohorts`.
- Why: it is the smallest scheduler-internal handoff slice that tests prompt-processing/prefill unification without broadening immediately into prompt-cache API changes, product-surface changes, or a full resident active batch object.
- Evidence: `src/mlxs/batch/scheduler.py`, `src/mlxs/product_surfaces/batched_serving.py`, installed `mlx_lm/generate.py::_process_prompts`, installed `mlx_lm/generate.py::_next`.

## D043 — 2026-04-19 — rejected

- Decision: reject the aligned-cohort resident prefill-to-completion handoff slice as a promotable AC2 redesign.
- Why: local AC2 truth-first validation preserved parity exactly, but `256` prompt throughput regressed materially and `2048` prompt throughput improved only slightly; that is too weak/mixed for remote-authoritative budget.
- Evidence: `/tmp/mlxs_m01_r7_worktree_local_ac2_20260419_1.json`, `/tmp/mlxs_m01_r7_main_local_ac2_20260419_1.json`.

## D044 — 2026-04-19 — accepted

- Decision: the next adjacent batch-side path is `M01.R8 integrated resident batch contract redesign ranking`.
- Why: both narrow batch-side entry slices are now rejected:
  1. resident active completion-batch object
  2. resident prefill-to-first-decode handoff
  The next useful move is a broader integrated batch-contract ranking rather than another local scheduler seam.
- Evidence: D040, D043, `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`, `src/mlxs/batch/scheduler.py`, installed `mlx_lm/generate.py::Batch`.

## D045 — 2026-04-19 — accepted

- Decision: advance from `R8` ranking to `M01.R9 integrated resident fast-batch contract implementation`.
- Why: the missing substrate is now explicit: MLXs has no batch-aware cache/runtime contract at all, while `mlx_lm` batch throughput depends on a resident `Batch` object plus batch-aware cache operations. More scheduler-local ranking would not add new information.
- Evidence: `src/mlxs/batch/scheduler.py`, `src/mlxs/protocols/cache.py`, installed `mlx_lm/generate.py::_make_cache`, installed `mlx_lm/generate.py::Batch`.

## D046 — 2026-04-19 — accepted

- Decision: the ranked implementation thesis is `private integrated resident fast-batch contract for the canonical AC2 fast path`.
- Why: the strongest remaining AC2 tax is not one more seam inside the scheduler; it is the absence of a resident batch owner that combines prompt processing, current token ownership, batch-aware cache mutation, in-place filtering, and per-row cache extraction.
- Evidence: D035-D044, `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`, `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`, `src/mlxs/batch/scheduler.py`, installed `mlx_lm/generate.py::_process_prompts`, installed `mlx_lm/generate.py::_next`, installed `mlx_lm/generate.py::Batch`.

## D047 — 2026-04-19 — accepted

- Decision: the fallback thesis is `shared Layer 1 + Layer 3 batch-capable progression redesign`.
- Why: if a private Layer 3 resident fast batch still fails to clear a strong AC2 gate, the remaining missing leverage is likely shared progression/materialization ownership, not another Layer 3-local reorganization.
- Evidence: `RESEARCH.md` ranked vs fallback theses, current `runtime_core.decode` / `runtime_core.greedy` contracts, D046.

## D048 — 2026-04-19 — accepted

- Decision: the first integrated redesign is restricted to the canonical AC2 fast path only.
- Why: v1 should optimize for the highest-value comparable surface first:
  - plain `KVCache`
  - text-only
  - greedy
  - no-logprobs
  - no processors
  - homogeneous cohorts
  Imported prompt-cache requests, `ArraysCache` / `RotatingKVCache` / `CacheList`, and featureful requests stay on the accepted legacy scheduler path.
- Evidence: `src/mlxs/protocols/cache.py`, `src/mlxs/cache/{kv.py,arrays.py,rotating.py,cache_list.py}`, `src/mlxs/product_surfaces/batched_serving.py`, installed `mlx_lm/generate.py::_make_cache`, D046.

## D049 — 2026-04-19 — accepted

- Decision: do not pursue for `R9`:
  1. more scheduler-local merge/scatter or handoff tweaks
  2. prompt-cache API redesign
  3. product-surface API changes
  4. `general_path` re-entry
  5. compile-first changes
  6. full mixed-cache-family batching in v1
- Why: these either already failed, are outside the canonical AC2 fast path, or would broaden the redesign without first proving the new resident batch substrate.
- Evidence: D031, D033, D040, D043, `FREEZE.md`, `RESEARCH.md`, `ARCHITECTURE_NOTES.md`.

## D050 — 2026-04-19 — accepted

- Decision: remote-authoritative batch validation for `R9` uses fresh remote mirrors of the accepted local baseline and the redesign worktree, not the older remote accepted tree.
- Why: the older remote tree lags the accepted local batch surface (`BatchScheduler` constructor and prompt-cache support), so a direct candidate-vs-old-tree compare would not be apples-to-apples for AC2.
- Evidence: live read of `/tmp/mlxs_layer1_candidate_20260417_1/src/mlxs/batch/scheduler.py` vs local accepted `src/mlxs/batch/scheduler.py`.

## D051 — 2026-04-19 — rejected

- Decision: reject `R9 private integrated resident fast-batch contract` as a promotable AC2 redesign.
- Why: local truth-first evidence was strongly positive, but the remote-authoritative compare split sharply by prompt length:
  - prompt `256`: strong win
  - prompt `2048`: strong regression
  Exact parity held, but the mixed remote result is too large to promote.
- Evidence: `/tmp/mlxs_m01_r9_worktree_local_ac2_20260419_1.json`, `/tmp/mlxs_m01_r9_main_local_ac2_20260419_1.json`, remote mirrors `/tmp/mlxs_m01_r9_candidate_remote_ac2_20260419_2.json`, `/tmp/mlxs_m01_r9_control_remote_ac2_20260419_2.json`.

## D052 — 2026-04-19 — accepted

- Decision: the next adjacent redesign path is `M01.R10 shared Layer 1 + Layer 3 batch-capable progression redesign ranking`.
- Why: the private Layer 3-only resident fast batch improved the short canonical surface but failed the long authoritative surface. The remaining leverage likely requires shared ownership changes across Layer 1 progression/materialization and Layer 3 batching, not another Layer 3-local variant.
- Evidence: D047 fallback thesis, D051 rejection, `src/mlxs/runtime_core/decode.py`, `src/mlxs/runtime_core/greedy.py`, `src/mlxs/batch/scheduler.py`, installed `mlx_lm/generate.py::{_step,Batch,_next}`.

## D053 — 2026-04-19 — accepted

- Decision: the post-`R9` diagnosis is that short-prompt wins came from removing Layer 3 reconstruction cost, while the long-prompt regression came from leaving prompt/decode progression and materialization ownership outside Layer 1.
- Why: `R9` changed Layer 3 only. It improved decode-dominated short prompts strongly, but long prompts regressed remotely on both TTFT and completion, which points to the missing shared owner for batched prefill, current/next token progression, and materialization boundaries.
- Evidence: `/tmp/mlxs_m01_r9_worktree_local_ac2_20260419_1.json`, `/tmp/mlxs_m01_r9_main_local_ac2_20260419_1.json`, `/tmp/mlxs_m01_r9_candidate_remote_ac2_20260419_2.json`, `/tmp/mlxs_m01_r9_control_remote_ac2_20260419_2.json`, `src/mlxs/runtime_core/{prefill.py,decode.py,greedy.py}`, `src/mlxs/batch/scheduler.py`.

## D054 — 2026-04-19 — accepted

- Decision: the ranked `R10` thesis is `shared Layer 1 batch-capable progression kernel consumed by Layer 3 orchestration`.
- Why: the strongest remaining leverage is to move batched prefill, prepared-step progression, materialization, and row-level cache filter/extract operations into Layer 1 while leaving queueing, row lifecycle, and product-near event emission in Layer 3 only.
- Evidence: D053, `docs/refactor/performance-core-spec.md`, `RESEARCH.md`, `ARCHITECTURE_NOTES.md`, installed `mlx_lm/generate.py::{_process_prompts,_step,_next,Batch}`.

## D055 — 2026-04-19 — accepted

- Decision: the fallback `R10` thesis is `Layer 1-owned batched prefill/materialization with Layer 3 decode ownership retained temporarily`.
- Why: if the full shared batch progression kernel is too risky, the next smaller integrated move is still to move prompt prefill and materialization boundaries into Layer 1 first, because those owners are the most implicated by the `R9` long-prompt regression.
- Evidence: D053, current `runtime_core.prefill`, current `batch.scheduler`, remote `R9` long-prompt regression.

## D056 — 2026-04-19 — accepted

- Decision: do not pursue for `R10`:
  1. another Layer 3-only resident batch owner
  2. another scheduler-local handoff/merge/scatter tweak
  3. prompt-cache API redesign
  4. product-surface API changes
  5. full mixed-cache-family support in the first slice
- Why: these either already failed, are outside the shared-ownership diagnosis, or would widen the redesign before the new Layer 1 batch progression contract is proven.
- Evidence: D040, D043, D051, `FREEZE.md`, D053-D055.

## D057 — 2026-04-19 — accepted

- Decision: the first integrated `R10` slice is `aligned plain-KV shared batch progression kernel`.
- Why: it is the smallest slice that genuinely changes shared Layer 1 + Layer 3 ownership:
  - Layer 1 owns batched prefill, prepared-step scheduling, materialization, and row-level cache filter/extract
  - Layer 3 owns row metadata, stop semantics, queue/admission, and legacy fallback
- Evidence: D054, D056, current `runtime_core.decode` helpers, current `batch.scheduler`, installed `mlx_lm.BatchGenerator`.

## D058 — 2026-04-19 — accepted

- Decision: promote `R10 aligned plain-KV shared batch progression kernel`.
- Why: the shared Layer 1 + Layer 3 slice cleared both gates:
  - local truth-first compare was strongly positive on both canonical AC2 prompt sizes with exact parity
  - remote-authoritative compare on fresh mirrors was positive on both prompt sizes with exact parity
- Evidence: local `/tmp/mlxs_m01_r10_worktree_local_ac2_20260419_1.json`, `/tmp/mlxs_m01_r10_main_local_ac2_20260419_1.json`; remote `/tmp/mlxs_m01_r10_candidate_remote_ac2_20260419_1.json`, `/tmp/mlxs_m01_r10_control_remote_ac2_20260419_1.json`.

## D059 — 2026-04-19 — accepted

- Decision: the scheduler-level AC2 probe is retained as neutral promoted tooling.
- Why: it is now the authoritative repo-native way to validate scheduler-level AC2 changes on local and remote mirrored baselines, replacing ad hoc `/tmp` scripts.
- Evidence: `benchmarks/mlxs_vs_mlx_lm/scheduler_ac2_probe.py`, D058.

## D060 — 2026-04-19 — accepted

- Decision: the next active path is `M01.R11 post-R10 acceptance closure rerank`.
- Why: `R10` is promoted, but AC2 absolute closure vs `mlx_lm` still needs to be re-evaluated from the new accepted state, and AC1 remains open. The next step is no longer implementation-by-default; it is reranking from the promoted baseline.
- Evidence: D058, `STATUS.md`, `WORKLOG.md`, existing AC1/AC2 acceptance state.

## D061 — 2026-04-19 — accepted

- Decision: create a durable git checkpoint commit for the accepted promoted baseline through `R10`.
- Why: the accepted main baseline now contains a long chain of promoted code and the full file-backed execution system; it needs a durable checkpoint before `R11` continues from it.
- Evidence: local main commit `fcb69de` (`checkpoint: capture accepted baseline through R10`).

## D062 — 2026-04-19 — accepted

- Decision: normalize `R11` reranking from durable checkpoint `fcb69de` using current accepted benchmark surfaces, not only pre-`R10` acceptance ratios.
- Why: `R10` materially changed the accepted batch path, so historical AC2 ratios alone are no longer sufficient for post-promotion ranking.
- Evidence: `STATUS.md`, `WORKLOG.md`, checkpoint `fcb69de`, current accepted code state.

## D063 — 2026-04-19 — accepted

- Decision: `AC1` remains open and unchanged after `R10`.
- Why: `R10` changed the batch path only. The current accepted single-request evidence still shows no accepted reference clearing `AC1 >= 1.10x decode`.
- Evidence: `STATUS.md` accepted reference set, existing accepted Class A remote artifacts (`Llama 1B`, `Qwen 1.5B`, `Llama 3B`), and the absence of single-request code changes in `R10`.

## D064 — 2026-04-19 — accepted

- Decision: `AC13` remains blocked and unchanged after `R10`.
- Why: the promoted `R10` batch kernel does not change the previously blocked eager-prefill / TTFT / RSS front.
- Evidence: `STATUS.md`, `RISKS.md`, prior accepted AC13 evidence (`best eager prefill 1.02808x < 1.05x`).

## D065 — 2026-04-19 — accepted

- Decision: `AC2` is the strongest next open target after `R10`, and the current accepted-surface direct rerun is materially below `mlx_lm`.
- Why: direct current rerun on the approved remote host using the promoted accepted batch path and a fresh `mlx_lm` batch baseline measured:
  - prompt `256 requests/s ~0.58268x`
  - prompt `2048 requests/s ~0.61619x`
  This makes AC2 clearly open and higher-leverage than AC1 or blocked AC13.
- Evidence: `/tmp/mlxs_m01_r10_candidate_remote_ac2_20260419_1.json`, `/tmp/mlxs_r11_mlx_lm_remote_ac2_20260419_2.json`.

## D066 — 2026-04-19 — accepted

- Decision: the next active path is `M01.R12 AC2 accepted-surface reconciliation and closure rerun`.
- Why: after `R10`, the immediate need is not another redesign slice; it is to reconcile the current accepted AC2 surface against the pre-`R10` ledger and close or precisely quantify the remaining gap before choosing another redesign front.
- Evidence: D062-D065, `STATUS.md`, `REPORT.md`, `TASKS.md`.

## D067 — 2026-04-19 — accepted

- Decision: the current direct AC2 accepted-surface rerun is the correct surface to use for post-`R10` acceptance ranking.
- Why: the direct `mlx_lm` rerun on the approved remote host produced stable absolute throughput close to the old authoritative comparator, while the promoted accepted MLXs surface remained far below it. That supports the current direct surface rather than the old AC2 ledger.
- Evidence: `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`, `/tmp/mlxs_m01_r10_candidate_remote_ac2_20260419_1.json`, `/tmp/mlxs_r11_mlx_lm_remote_ac2_20260419_2.json`.

## D068 — 2026-04-19 — accepted

- Decision: the old pre-`R10` AC2 ledger is superseded for current accepted-surface ranking.
- Why: it measured an older accepted MLXs surface that no longer matches the current durable baseline `fcb69de` + promoted `R10` code. It remains historical evidence, but not the normalized acceptance ledger for the current baseline.
- Evidence: D062, D065, D067, `STATUS.md`, `WORKLOG.md`.

## D069 — 2026-04-19 — accepted

- Decision: the next active path is `M01.R13 AC2 accepted-surface provenance audit and ledger rewrite`.
- Why: `R12` established AC2 truth well enough to stop treating the old ledger as authoritative, but the provenance of the accepted-surface drift still needs to be written down before another redesign family is justified.
- Evidence: D067-D068, `STATUS.md`, `REPORT.md`, `TASKS.md`.

## D070 — 2026-04-19 — accepted

- Decision: normalize the post-`R14` accepted ledger around the repaired AC2 direct comparator, not the stale `R11/R12/R13` collapse ledger.
- Why: the authoritative direct comparator on the approved remote host now shows:
  - prompt `256 requests/s ~0.99661x`
  - prompt `2048 requests/s ~1.09404x`
  with exact output parity on both prompt sizes. That supersedes the stale catastrophic AC2 ranking.
- Evidence: `/tmp/mlxs_ac2_direct_compare_candidate_remote_20260419_2.json`.

## D071 — 2026-04-19 — accepted

- Decision: rerank the open targets from the current accepted state as:
  1. `AC1`
  2. narrow `AC2` prompt-`256` closure only if needed later
  3. `AC13` blocked-for-now
- Why: `AC1` still misses its actual acceptance bar on every accepted reference, while `AC2` is already positive on prompt `2048` and only near-parity on prompt `256`.
- Evidence: accepted Class A artifacts `/tmp/mlxs_class_a_accepted_remote_20260418_1.json`, `/tmp/mlxs_class_a_qwen15b_promoted_remote_20260418_3.json`, `/tmp/mlxs_class_a_llama32_3b_promoted_remote_20260418_1.json`; direct AC2 comparator `/tmp/mlxs_ac2_direct_compare_candidate_remote_20260419_2.json`; surviving AC13 file-backed value `1.02808x < 1.05x`.

## D072 — 2026-04-19 — accepted

- Decision: the authoritative AC1 ledger remains the accepted remote Class A artifacts from `2026-04-18` for current reranking.
- Why: reduced-run exploratory `R16` reruns are useful truth-first guardrails, but they diverged sharply on `Qwen 256` and `Llama 3B 256` and are not strong enough by themselves to rewrite accepted AC1 status.
- Evidence: `/tmp/mlxs_class_a_accepted_remote_20260418_1.json`, `/tmp/mlxs_class_a_qwen15b_promoted_remote_20260418_3.json`, `/tmp/mlxs_class_a_llama32_3b_promoted_remote_20260418_1.json`, plus truth-first guardrails `/tmp/mlxs_m01_r16_llama1b_remote_class_a_20260419_1.json`, `/tmp/mlxs_m01_r16_qwen256_remote_class_a_20260419_1.json`, `/tmp/mlxs_m01_r16_llama3b256_remote_class_a_20260419_1.json`.

## D073 — 2026-04-19 — accepted

- Decision: `R16` shows the benchmark/core split remains materially open on short-prompt AC1 cases after `R2`, but not on the long-prompt `Llama-3.2-1B 2048` case.
- Why: local truth-first helper/core comparison measured:
  - `Llama-3.2-1B 256 core/helper decode ratio ~0.8212`
  - `Qwen2.5-1.5B 256 core/helper decode ratio ~0.7911`
  - `Llama-3.2-1B 2048 core/helper decode ratio ~1.0277`
  Token counts matched in all local cases.
- Evidence: `/tmp/mlxs_m01_r16_local_core_split_20260419_1.json`.

## D074 — 2026-04-19 — accepted

- Decision: the next active path is `M01.R17 AC1 short-prompt helper/core convergence slice definition`.
- Why: the strongest current single-request leverage is no longer broad core unification in the abstract; it is a narrow shared inline short-prompt progression slice that converges the benchmark helper and `runtime_core.run_greedy` on the weakest accepted AC1 surface.
- Evidence: D071-D073, `RESEARCH.md` benchmark/core split thesis, current `benchmarks/mlxs_vs_mlx_lm/backends.py`, current `src/mlxs/runtime_core/greedy.py`.

## D075 — 2026-04-19 — accepted

- Decision: tighten `R17` to the exact first bounded slice `AC1 short-prompt materialization contract alignment`.
- Why: a no-patch local probe showed that removing grouped `next_prepared` eval from the short-prompt `run_greedy` materialization path already moves the weakest cases in the right direction:
  - `Llama-3.2-1B 256 no-group/current decode ratio ~1.10696x`
  - `Qwen2.5-1.5B 256 no-group/current decode ratio ~1.09965x`
  Token counts and token prefixes matched on both cases.
- Evidence: `/tmp/mlxs_m01_r17_no_group_materialize_probe_local_20260419_1.json`, current `benchmarks/mlxs_vs_mlx_lm/backends.py`, current `src/mlxs/runtime_core/greedy.py`.

## D076 — 2026-04-19 — rejected

- Decision: reject `R17 run_greedy-only short-prompt materialization contract alignment` as a promotable AC1 slice.
- Why:
  - local truth-first comparison against the exact pre-patch `run_greedy` behavior was positive on the intended short-prompt cases
  - the authoritative remote AC1 surface did not produce a clean accepted-surface win on the weakest accepted case
  - the deeper issue is structural: a `run_greedy`-only patch does not directly move the current accepted AC1 helper benchmark surface
- Evidence: `/tmp/mlxs_m01_r17_local_compare_20260419_1.json`, `/tmp/mlxs_m01_r17_llama1b_candidate_remote_20260419_1.json`, `/tmp/mlxs_m01_r17_qwen256_candidate_remote_20260419_1.json`, `/tmp/mlxs_m01_r17_llama3b256_candidate_remote_20260419_1.json`, current `benchmarks/mlxs_vs_mlx_lm/backends.py`.

## D077 — 2026-04-19 — accepted

- Decision: the next active path is `M01.R18 AC1 shared short-prompt helper/core convergence slice definition`.
- Why: the program still needs to close the short-prompt helper/core split, but the failure of `R17` shows the next slice must be shared by both the benchmark helper and `runtime_core.run_greedy` so it can move the accepted AC1 surface and the real core together.
- Evidence: D073, D076, `RESEARCH.md` benchmark/core split thesis, current `benchmarks/mlxs_vs_mlx_lm/backends.py`, current `src/mlxs/runtime_core/greedy.py`.

## D078 — 2026-04-20 — rejected

- Decision: reject `R18 shared short-prompt helper/core convergence slice` as a promotable AC1 slice.
- Why:
  - helper/core convergence remained directionally plausible
  - but the shared per-token generator loop regressed the weakest accepted short-prompt case too sharply in local truth-first validation
  - that made remote validation unjustified
- Evidence: `/tmp/mlxs_m01_r18_local_surface_20260419_candidate.json`, `/tmp/mlxs_m01_r18_local_surface_20260419_baseline.json`.

## D079 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R19 AC1 low-overhead inline short-prompt convergence ranking`.
- Why: `R18` proved the shared family is still directionally right, but the next slice had to preserve helper/core alignment without the per-token generator overhead.
- Evidence: D076, D078, current `benchmarks/mlxs_vs_mlx_lm/backends.py`, current `src/mlxs/runtime_core/{decode.py,greedy.py}`.

## D080 — 2026-04-20 — rejected

- Decision: reject `R19 low-overhead inline short-prompt helper/core convergence` as a promotable AC1 slice.
- Why:
  - local correctness and local truth-first evidence were strong
  - but the decisive authoritative remote weakest case still failed to improve:
    - control `Llama-3.2-1B 256 eager decode ~1.00214x`
    - candidate `Llama-3.2-1B 256 eager decode ~0.99967x`
  - after `R17`, `R18`, and `R19`, the convergence family is no longer implementation-ready enough to justify another patch by inertia
- Evidence: `/tmp/mlxs_m01_r19_llama1b_control_remote_20260420_1.json`, `/tmp/mlxs_m01_r19_llama1b_candidate_remote_20260420_1.json`, `/tmp/mlxs_m01_r19_local_surface_20260420_candidate.json`, `/tmp/mlxs_m01_r18_local_surface_20260419_baseline.json`.

## D081 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R20 AC1 weakest-case helper-surface hot-loop decomposition and rerank`.
- Why: the next useful step is no longer another helper/core convergence patch; it is to decompose the real accepted weakest-case helper surface and rerank the next non-exhausted AC1 family from evidence.
- Evidence: D076, D078, D080, accepted AC1 artifacts, and the repeated failure of convergence variants to move the decisive weakest-case remote surface.

## D082 — 2026-04-20 — accepted

- Decision: for the current benchmark-relevant `Llama` and `Qwen` references, implementation mismatch is not currently the leading explanation for the open AC1 gap.
- Why:
  - `Llama-3.2-1B` focused audit showed matching config interpretation, matching attention metadata/projection shapes, exact controlled prompt prefill/decode logit parity, exact first-layer output parity, and matching cache offsets vs `mlx_lm`
  - `Qwen2.5-1.5B` focused sanity check showed matching resolved attention metadata and exact controlled prompt prefill/decode logit/cache-offset parity vs `mlx_lm`
- Evidence: `docs/execution/milestones/M01-performance-architecture-redesign/MODEL_AUDIT.md`.

## D083 — 2026-04-20 — accepted

- Decision: `R20` decomposition identifies helper-path `schedule_next` / forward-build cost as the dominant remaining weakest-case bottleneck.
- Why: on the authoritative weakest accepted surface `mlx-community/Llama-3.2-1B-Instruct-4bit` with prompt `256`, remote decomposition measured:
  - MLXs `schedule_next_per_token_s ~0.00302744`
  - `mlx_lm schedule_next_per_token_s ~0.00034372`
  - ratio `~8.81x`
  Meanwhile MLXs selection/materialization per token was lower, not higher:
  - MLXs `~0.00206306`
  - `mlx_lm ~0.00397474`
  - ratio `~0.52x`
- Evidence: `/tmp/mlxs_m01_r20_helper_hotloop_remote_20260420_1.json`.

## D084 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R21 AC1 helper-surface schedule-next forward-build contract redesign`.
- Why: after `R20`, the next non-exhausted AC1 family is the weakest-case helper short-prompt lookahead scheduling contract itself; it now has a clear bottleneck, clean model-correctness classification, and higher leverage than another broad convergence experiment.
- Evidence: D082, D083, accepted AC1 ledger, and `R17`/`R18`/`R19` rejection record.

## D085 — 2026-04-20 — rejected

- Decision: reject `R22 resident token-lookahead helper-contract redesign` as a promotable AC1 slice.
- Why:
  - the redesign removed prepared-step/result-object churn from the benchmark lookahead loop and moved the helper closer to a resident `mlx_lm`-style token loop
  - local control/candidate evidence was too small and mixed:
    - `Llama-3.2-1B 256 candidate/control decode ratio ~1.00480x`
    - `Llama-3.2-1B 2048 candidate/control decode ratio ~0.98952x`
  - authoritative remote control/candidate evidence stayed slightly regressive in absolute MLXs decode throughput:
    - `Llama-3.2-1B 256 candidate/control decode ratio ~0.99904x`
    - `Llama-3.2-1B 2048 candidate/control decode ratio ~0.99686x`
  - the higher candidate `mlxs/mlx_lm` ratios on the remote rerun came from baseline drift in the paired `mlx_lm` run, not from a stronger MLXs result
- Evidence: `/tmp/mlxs_m01_r22_control_local_20260420_1.json`, `/tmp/mlxs_m01_r22_candidate_local_20260420_2.json`, `/tmp/mlxs_m01_r22_control_remote_20260420_1.json`, `/tmp/mlxs_m01_r22_candidate_remote_20260420_1.json`.

## D086 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R23 AC1 model/cache forward-step substrate parity audit`.
- Why:
  - `R22` shows that simplifying helper token choreography is no longer the dominant lever
  - the open weakest-case tax survives a near-`mlx_lm` resident benchmark loop, which pushes the next investigation below the helper contract
  - quick current-state read confirms MLXs and `mlx_lm` still diverge on the concrete forward-step substrate in the weakest-case family:
    - MLXs loads `mlxs.models.llama.Model` with `mlxs.cache.kv.KVCache`
    - `mlx_lm` loads `mlx_lm.models.llama.Model` with `mlx_lm.models.cache.KVCache`
  - the next useful evidence must isolate forward-step, cache-update, and adjacent mask/attention setup cost before any new patch
- Evidence: D083-D085, `src/mlxs/models/llama.py`, `src/mlxs/cache/kv.py`, local type reads from `mlxs.load.loader.load_model(...)` and `mlx_lm.load(...)`, plus the rejected `R22` local/remote artifacts.

## D087 — 2026-04-20 — accepted

- Decision: `R23` starts with an explicit live audit artifact and a hard local evidence gate before any substrate patch.
- Why:
  - `R22` already proved that helper-only redesign is exhausted enough that the next patch cannot be justified by intuition alone
  - early no-patch substrate reads show a narrower target than broad model-wrapper redesign:
    - simple steady-state cached forward stayed near parity (`MLXs ~0.01273s`, `mlx_lm ~0.01267s`)
    - simple first-decode-after-prefill read stayed directionally worse for MLXs (`~0.01480s` vs `~0.00989s`)
    - isolated synthetic first-call `KVCache.update_and_fetch()` asymmetry did not survive order reversal
  - that combination points the next investigation at the prompt-tail to first-decode boundary, not at the whole steady-state model forward path
- Evidence: `docs/execution/milestones/M01-performance-architecture-redesign/R23_SUBSTRATE_AUDIT.md`, `src/mlxs/models/llama.py`, `src/mlxs/cache/kv.py`, installed `mlx_lm.models.{llama,cache}`, and local no-patch probe reads on `Llama-3.2-1B 256`.

## D088 — 2026-04-20 — rejected

- Decision: reject a bounded single-request `R23` first-decode substrate patch as the next implementation slice.
- Why:
  - local and remote interleaved substrate probes both showed the same structure:
    - current MLXs total prompt-tail / first-decode boundary was already better than comparable `mlx_lm`
    - the first no-patch integrated boundary variant was materially worse than current MLXs
  - the isolated `first_decode_forward` component remained slower in MLXs, but the full boundary did not identify a bounded patch that actually improves the decisive surface
  - that fails the `R23` evidence gate
- Evidence: `/tmp/mlxs_m01_r23_forward_step_probe_local_20260420_1.json`, `/tmp/mlxs_m01_r23_forward_step_probe_remote_20260420_1.json`, `docs/execution/milestones/M01-performance-architecture-redesign/R23_SUBSTRATE_AUDIT.md`.

## D089 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R24 AC2 batch-first resident progression redesign rerank`.
- Why:
  - `R23` closed the adjacent single-request front negatively
  - the original batch-first fallback thesis is now the next strongest adjacent path
  - the repaired direct AC2 comparator still leaves prompt `256` near-parity while prompt `2048` is already positive, which is the right shape for a new batch redesign ranking
- Evidence: D071, D083, D088, `/tmp/mlxs_ac2_direct_compare_candidate_remote_20260419_2.json`, `docs/execution/milestones/M01-performance-architecture-redesign/R24_BATCH_FIRST_RERANK.md`.

## D090 — 2026-04-20 — accepted

- Decision: rerank the current batch structural taxes as:
  1. resident prompt-processing / prefill-to-decode handoff
  2. extendable active batch ownership
  3. cache substrate breadth for canonical continuous residency
  4. scheduler-level admission/prompt/decode/event coupling
  5. event/materialization placement as incidental only
- Why:
  - repaired direct comparator shows prompt `256` is the only open AC2 closure target while prompt `2048` is already positive
  - `R9` proved Layer 3-only residency can improve short prompts while breaking long prompts
  - `R10` proved shared Layer 1 + Layer 3 progression ownership is real leverage and already fixed much of the long-prompt surface
  - current code still keeps resident batching conditional and split between `_SharedFastBatch` and the legacy scheduler path, while `mlx_lm` keeps resident `PromptProcessingBatch` and `GenerationBatch` owners
- Evidence: `/tmp/mlxs_ac2_direct_compare_candidate_remote_20260419_2.json`, `/tmp/mlxs_ac2_scheduler_baseline_remote_20260418_1.json`, `/tmp/mlxs_ac2_scheduler_decomp_remote_20260418_1.json`, `/tmp/mlxs_m01_r9_candidate_remote_ac2_20260419_2.json`, `/tmp/mlxs_m01_r9_control_remote_ac2_20260419_2.json`, `/tmp/mlxs_m01_r10_candidate_remote_ac2_20260419_1.json`, `/tmp/mlxs_m01_r10_control_remote_ac2_20260419_1.json`, `src/mlxs/batch/scheduler.py`, `src/mlxs/batch/fast_batch.py`, `src/mlxs/runtime_core/batch_progression.py`, installed `mlx_lm.generate::{BatchGenerator,PromptProcessingBatch,GenerationBatch}`.

## D091 — 2026-04-20 — accepted

- Decision: the ranked `R24` redesign thesis is `canonical resident prompt-batch + extendable active-batch ownership over shared Layer 1 batch progression`.
- Why:
  - it directly attacks the remaining prompt-`256` fixed-cost / handoff tax
  - it preserves the `R10` lesson that Layer 1 progression ownership is the right substrate
  - it avoids reopening the rejected Layer 3-only `R9` family and the rejected narrow `R6`/`R7` batch entry slices
- Evidence: D090, current batch code, and `docs/execution/milestones/M01-performance-architecture-redesign/R24_BATCH_DIVERGENCE_MAP.md`.

## D092 — 2026-04-20 — accepted

- Decision: the `R24` fallback thesis is `bounded resident prompt-prefill handoff into an extendable fast batch owner`.
- Why: if the full dual-owner redesign is too broad, the smallest still-coherent fallback is to make the resident fast batch extendable at the prompt→generation boundary without dropping back to per-sequence scheduler state.
- Evidence: D090-D091, current `_SharedFastBatch`, current `BatchScheduler`, and upstream `PromptProcessingBatch.generate()` + `GenerationBatch.extend()`.

## D093 — 2026-04-20 — accepted

- Decision: `R24-S1` shows the first bounded batch slice should move from prompt/handoff ownership to the active-batch first generation step.
- Why:
  - local and remote no-patch decomposition both showed the same shape:
    - MLXs fast-path activation was already competitive or better than the upstream prompt/generation activation
    - MLXs first active-batch generation step was stably slower
  - that makes prompt-batch ownership a thesis-level concern but not the first bounded implementation slice
- Evidence: `/tmp/mlxs_m01_r24_fast_batch_decomp_local_20260420_1.json`, `/tmp/mlxs_m01_r24_fast_batch_decomp_remote_20260420_1.json`, `docs/execution/milestones/M01-performance-architecture-redesign/R24_BATCH_DIVERGENCE_MAP.md`.

## D094 — 2026-04-20 — rejected

- Decision: reject the bounded `R24-S2` materialization-only active-batch slice as non-promotable.
- Why:
  - remote first-step decomposition showed the candidate improved the isolated micro surface strongly:
    - total `candidate/control ~0.45550x`
    - `materialize_s candidate/control ~0.14887x`
  - but the authoritative remote scheduler guardrail rejected it on the real open target:
    - prompt `256 candidate/control requests_per_s ~0.95880x`
    - prompt `256` TTFT and completion both regressed
  - this means the grouped current+next materialization is carrying useful overlap/readiness that the isolated micro probe did not capture
- Evidence: `/tmp/mlxs_m01_r24_fast_batch_first_step_remote_20260420_1.json`, `/tmp/mlxs_m01_r24_fast_batch_first_step_candidate_remote_20260420_2.json`, `/tmp/mlxs_m01_r24_control_remote_scheduler_20260420_1.json`, `/tmp/mlxs_m01_r24_candidate_remote_scheduler_20260420_1.json`.

## D095 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R24-S3 active-batch overlap decomposition`.
- Why:
  - `R24-S2` proved the first-step materialization call is not independent; it is coupled to downstream readiness across steps
  - the next missing evidence is therefore no longer inside one call site but inside the overlap contract between consecutive active-batch steps
- Evidence: D094 and `docs/execution/milestones/M01-performance-architecture-redesign/R24_BATCH_DIVERGENCE_MAP.md`.

## D096 — 2026-04-20 — accepted

- Decision: the initial `R24-S3` local overlap readback supports treating overlap/readiness as the next bounded evidence target.
- Why:
  - MLXs step 0 remains slower because of materialization
  - MLXs step 1 becomes faster than the comparable upstream core step
  - that is the expected shape when grouped current+next materialization is buying second-step readiness
- Evidence: `/tmp/mlxs_m01_r24_fast_batch_overlap_local_20260420_1.json`, `docs/execution/milestones/M01-performance-architecture-redesign/R24_BATCH_DIVERGENCE_MAP.md`.

## D097 — 2026-04-20 — accepted

- Decision: `R24-S3` is classified as `broader active-batch state-lifetime redesign`, not as a narrow overlap fix.
- Why:
  - local and remote overlap probes share the same directional shape:
    - step 0 remains materially slower for MLXs
    - step 1 scheduling/core becomes very strong for MLXs
    - step 1 end-to-end is near-parity remotely
  - that means grouped current+next materialization is not just paying for one overlap seam; it is part of how active-batch state carries readiness across the step boundary
- Evidence: `/tmp/mlxs_m01_r24_fast_batch_overlap_local_20260420_1.json`, `/tmp/mlxs_m01_r24_fast_batch_overlap_remote_20260420_1.json`, `docs/execution/milestones/M01-performance-architecture-redesign/R24_BATCH_DIVERGENCE_MAP.md`.

## D098 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R24-S4 active-batch state-lifetime redesign`.
- Why:
  - `R24-S3` ended decisively and does not justify another seam-level overlap tweak
  - the first bounded slice should redesign the resident dual-step state owner inside `_SharedFastBatch`, not just edit one eval/materialization call
- Evidence: D097 and `docs/execution/milestones/M01-performance-architecture-redesign/R24_ACTIVE_BATCH_STATE_LIFETIME.md`.

## D099 — 2026-04-20 — rejected

- Decision: reject the first concrete `R24-S4` candidate `R24-S4a resident dual-step _SharedFastBatch owner`.
- Why:
  - the candidate was already attempted locally and reverted
  - it made the fast-path activation and overlap probes materially worse
  - keeping `R24-S4` active under the same thesis/name would be an invalid retry of the same shape
- Evidence: current `R24` worktree readback plus local truth-first candidate artifacts cited in the prior session outcome.

## D100 — 2026-04-20 — accepted

- Decision: choose Outcome B.
- Why:
  - the broader batch family is not exhausted
  - but the first concrete `R24-S4a` shape is already rejected and frozen
  - the next valid move must therefore use a new identifier and a materially different thesis
- Evidence: D097-D099 and current file-backed inconsistency report.

## D101 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R25 canonical prompt-batch + extendable active-batch ownership redesign`.
- Why:
  - it changes the owner boundary itself instead of retrying state lifetime tweaks inside the old narrow `_SharedFastBatch`
  - it is materially different from `R24-S4a`
  - it returns to the broader `R24` architectural thesis with a new concrete shape
- Evidence: `docs/execution/milestones/M01-performance-architecture-redesign/R25_CANONICAL_BATCH_OWNERSHIP.md`, `R24_BATCH_DIVERGENCE_MAP.md`, and the rejected `R24-S4a` candidate.

## D102 — 2026-04-20 — rejected

- Decision: reject the `R25` canonical extendable active-owner family as a promotable redesign on the current canonical prompt-`256` scheduler surface.
- Why:
  - the first integrated token-resident extendable owner over per-row-offset batch cache regressed MLXs absolute scheduler throughput materially:
    - requests/s `candidate/control ~0.79339x`
    - p50 TTFT `~1.06247x`
    - p95 completion `~1.31978x`
  - the follow-up dual-ready extendable owner regressed further:
    - requests/s `candidate/control ~0.73350x`
    - p50 TTFT `~1.02370x`
    - p95 completion `~1.36299x`
  - that makes the current canonical aligned fast path the wrong place to inject per-row-offset extendability by default
- Evidence:
  - `/tmp/mlxs_m01_r25_scheduler_control_local_20260420_1.json`
  - `/tmp/mlxs_m01_r25_scheduler_candidate_local_20260420_1.json`
  - `/tmp/mlxs_m01_r25_scheduler_candidate_local_20260420_2.json`

## D103 — 2026-04-20 — accepted

- Decision: the repaired direct AC2 canonical compare is not sufficient as the sole ranking surface for late-admission / extendable batch redesign.
- Why:
  - it measures a static aligned cohort where extension never occurs
  - the new staggered local compare exercises exactly the surface the rejected `R25` thesis was trying to improve and shows a much larger practical gap:
    - `mlxs/mlx_lm requests_per_s ~0.69428x`
    - p50 TTFT `~1.08180x`
    - p95 completion `~1.37731x`
    - exact output parity `True`
- Evidence: `/tmp/mlxs_m01_r25_staggered_local_20260420_1.json`

## D104 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R26 dual-owner batch architecture rerank on staggered extension surface`.
- Why:
  - current evidence says the aligned plain-KV fast batch should stay hot on the current static canonical AC2 surface
  - current evidence also says dynamic late admission remains a large practical gap versus `mlx_lm`
  - the coherent next move is to separate those concerns rather than forcing extendability into the aligned hot path blindly
- Evidence:
  - D102
  - D103
  - `docs/execution/milestones/M01-performance-architecture-redesign/R26_DUAL_OWNER_BATCH_RERANK.md`

## D105 — 2026-04-20 — accepted

- Decision: establish the staggered late-admission surface remotely before ranking the next redesign slice.
- Why:
  - the accepted remote tree did not match file-backed assumptions, so a fresh isolated remote baseline had to be created from local `fcb69de` truth
  - the resulting authoritative staggered compare confirms the current accepted MLXs baseline is still materially behind on that surface:
    - `mlxs/mlx_lm requests_per_s ~0.85088x`
    - exact output parity `True`
- Evidence:
  - `/tmp/mlxs_m01_r26_staggered_remote_20260420_1.json`
  - remote sync path `/tmp/mlxs_m01_r26_remote_baseline_20260420_1`

## D106 — 2026-04-20 — accepted

- Decision: the decisive staggered-surface tax is late-request starvation before continuation, not initial-request TTFT.
- Why:
  - authoritative remote decomposition shows:
    - MLXs late request first-token step `129`
    - `mlx_lm` late request first-token step `4`
    - MLXs late request completion step `255`
    - `mlx_lm` late request completion step `131`
  - initial-request step counts are already aligned
- Evidence:
  - `/tmp/mlxs_m01_r26_staggered_decomp_remote_20260420_1.json`
  - local corroboration `/tmp/mlxs_m01_r26_staggered_decomp_local_20260420_1.json`

## D107 — 2026-04-20 — rejected

- Decision: reject `R26-S1 full dynamic late-admission owner` as a promotable redesign slice.
- Why:
  - local staggered control/candidate compare regressed throughput materially:
    - requests/s `candidate/control ~0.77448x`
    - p95 completion `~1.34605x`
  - exact output parity also broke
- Evidence:
  - `/tmp/mlxs_m01_r26_staggered_control_local_20260420_1.json`
  - `/tmp/mlxs_m01_r26_staggered_candidate_local_20260420_1.json`

## D108 — 2026-04-20 — rejected

- Decision: reject `R26-S2 concurrent late prefill plus legacy continuation handoff` as a promotable slice.
- Why:
  - it fixed the semantics strongly:
    - late first-token step `129 -> 2`
    - late completion step `255 -> 128`
  - but it still missed the fixed local gate on wall-clock performance:
    - requests/s `candidate/control ~0.99397x`
    - p95 completion `~1.04957x`
    - exact output parity `True`
  - that means synchronous full prompt work inside the main scheduler step is still too expensive
- Evidence:
  - `/tmp/mlxs_m01_r26b_staggered_control_local_20260420_1.json`
  - `/tmp/mlxs_m01_r26b_staggered_candidate_local_20260420_1.json`
  - `/tmp/mlxs_m01_r26b_staggered_decomp_candidate_local_20260420_1.json`

## D109 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R26-S3 bounded prompt-owner overlap redesign`.
- Why:
  - `R26-S2` proves prompt-side admission semantics are correct
  - `R26-S2` also proves the remaining loss is the synchronous whole-prompt cost inside the main scheduler step

## D110 — 2026-04-20 — accepted

- Decision: normalize the next runtime path name to `M01.R26-S4 integrated prompt-owner budgeted progression redesign`.
- Why:
  - `R26-S3` already served as the diagnosis/rerank block
  - the next work is a new implementation block and should not look like an ambiguous continuation of the old one
- Evidence:
  - `docs/execution/milestones/M01-performance-architecture-redesign/R26_STAGGERED_SURFACE_DIAGNOSIS.md`
  - `docs/execution/milestones/M01-performance-architecture-redesign/R26_S4_PROMPT_OWNER_BUDGETED_PROGRESSION.md`

## D111 — 2026-04-20 — rejected

- Decision: reject `R26-S4 integrated prompt-owner budgeted progression redesign` as a promotable slice.
- Why:
  - the candidate preserved exact parity and fixed the late request step positions to near-upstream:
    - late first-token step `4`
    - late completion step `130` vs `mlx_lm 131`
  - but the wall-clock staggered surface still missed the fixed local gate:
    - requests/s `candidate/control ~0.91001x`
    - p50 TTFT `~0.99142x`
    - p95 completion `~1.14951x`
  - that means Layer 3 prompt-owner budgeting plus dedicated-stream overlap is still not enough
- Evidence:
  - `/tmp/mlxs_m01_r26s4_staggered_control_local_20260420_1.json`
  - `/tmp/mlxs_m01_r26s4_staggered_candidate_local_20260420_1.json`
  - `/tmp/mlxs_m01_r26s4b_staggered_decomp_candidate_local_20260420_1.json`

## D112 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R26-S5 true-overlap prompt-progress substrate redesign`.
- Why:
  - `R26-S4` proves the prompt-owner semantics are already good enough
  - `R26-S4` also proves the remaining loss sits below that owner in the prompt-progress substrate / real overlap behavior
  - the next useful move must therefore target lower-level prompt-progress cost rather than retry another scheduler-level budget variant
- Evidence:
  - D111
  - `docs/execution/milestones/M01-performance-architecture-redesign/R26_S5_TRUE_OVERLAP_PROMPT_PROGRESS.md`

## D113 — 2026-04-20 — rejected

- Decision: reject `R26-S5 true-overlap prompt-progress substrate redesign` as a promotable slice.
- Why:
  - the candidate preserved exact parity and preserved the corrected late-request step positions:
    - late first-token step `4`
    - late completion step `130`
  - but it still failed to convert those semantics into a strong enough throughput win:
    - requests/s `candidate/control ~1.00040x`
    - p50 TTFT `~1.04423x`
    - p95 completion `~1.04108x`
  - that is not a strong enough lever for another descent in the same family
- Evidence:
  - `/tmp/mlxs_m01_r26s5_staggered_control_local_20260420_1.json`
  - `/tmp/mlxs_m01_r26s5_staggered_candidate_local_20260420_1.json`
  - `/tmp/mlxs_m01_r26s5_staggered_decomp_candidate_local_20260420_1.json`

## D114 — 2026-04-20 — accepted

- Decision: close the `R26` batch prompt-progress family for this phase and move to `M01.R27 global rerank after closing the R26 batch prompt-progress family`.
- Why:
  - the family now has multiple rejected shapes spanning scheduler-layer, owner-boundary, and lower-layer prompt-progress substrate variants
  - the surviving positive lesson is semantic only, not throughput-promotable
  - continuing in the same family would be churn rather than disciplined redesign
- Evidence:
  - D107
  - D108
  - D111
  - D113
  - `docs/execution/milestones/M01-performance-architecture-redesign/R27_POST_R26_GLOBAL_RERANK.md`

## D115 — 2026-04-20 — accepted

- Decision: `R27` is complete and the next active path is `M01.R28 integrated AC1 canonical single-request progression contract refactor`.
- Why:
  - `AC1` remains the strongest open target on the accepted reference set
  - the `AC1` front is already sufficiently understood
  - it has already exhausted enough bounded families that another mini-slice is no longer justified
  - the correct next move is therefore an integrated refactor rather than another helper/core local variant
- Evidence:
  - accepted `AC1` ledger in `STATUS.md`
  - `R17`/`R18`/`R19`/`R22`/`R23` rejection record
  - `docs/execution/milestones/M01-performance-architecture-redesign/R28_AC1_CANONICAL_PROGRESS_CONTRACT_REFACTOR.md`

## D116 — 2026-04-20 — rejected

- Decision: reject `R28 integrated AC1 canonical single-request progression contract refactor` as a promotable slice.
- Why:
  - the integrated refactor cleanly removed benchmark-local progression duplication
  - but it regressed or stayed neutral on the real accepted-sensitive local surfaces:
    - `Llama-3.2-1B 256 decode ~0.98852x`
    - `Llama-3.2-1B 2048 decode ~0.99909x`
    - `Qwen2.5-1.5B 256 decode ~0.90711x`
    - `Qwen2.5-1.5B 2048 decode ~0.92558x`
  - that means the remaining AC1 tax is not likely to be solved by another contract-level duplication cleanup in the same family
- Evidence:
  - `/tmp/mlxs_m01_r28_control_local_20260420_2.json`
  - `/tmp/mlxs_m01_r28_candidate_local_20260420_2.json`

## D117 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R29 AC1 lower-level substrate and capability rerank`.
- Why:
  - `AC1` remains strongest
  - helper/core micro-families, batch prompt-progress family, and contract-level canonical progression refactor are all now closed or rejected
  - current official MLX capability surface must now be reconsidered explicitly before the next AC1 implementation move
- Evidence:
  - D116
  - `docs/execution/milestones/M01-performance-architecture-redesign/R29_AC1_SUBSTRATE_RERANK.md`

## D118 — 2026-04-20 — accepted

- Decision: close the current `AC1` front for this phase after `R29`.
- Why:
  - the no-patch accepted-sensitive local substrate probe preserved exact parity and showed overall decode near parity / mixed
  - lower-boundary component deltas were mixed enough that no single lower-level internal or substrate-level bottleneck dominated across the surviving AC1 cases
  - current official MLX capabilities remain in the design space, but neither a built-in MLX path nor a custom extension / custom Metal kernel path is justified for `AC1` from this evidence
- Evidence:
  - `/tmp/mlxs_m01_r29_substrate_local_20260420_1.json`
  - current official MLX release/docs (`v0.31.1`, streams, custom extensions, custom Metal kernels)
  - `docs/execution/milestones/M01-performance-architecture-redesign/R29_AC1_SUBSTRATE_RERANK.md`

## D119 — 2026-04-20 — accepted

- Decision: the next active path is `M01.R30 AC2 lower-level substrate and capability rerank on the staggered batch surface`.
- Why:
  - after `R29`, the strongest remaining active open front is no longer `AC1`
  - `AC2` remains materially behind on the authoritative staggered late-admission surface
  - the prior prompt-owner / prompt-progress batch family is closed, so the next valid move is a lower-level rerank below that family rather than another batch micro-variant
- Evidence:
  - D118
  - `/tmp/mlxs_m01_r26_staggered_remote_20260420_1.json`
  - `/tmp/mlxs_m01_r26_staggered_decomp_remote_20260420_1.json`

## D120 — 2026-04-20 — accepted

- Decision: apply a repo-level process correction for MLXs runtime/performance work before any further technical implementation.
- Why:
  - the previous process became too slice-fragmented
  - too many bounded families and mini-refactors were being generated after fronts were already sufficiently understood
  - future work needs a binding rule that forces integrated refactor or closure/rerank once a front has burned about `2-3` bounded families without a strong win
- Evidence:
  - the accumulated `R17` through `R29` rejection chain
  - file-backed execution history in `STATUS.md`, `FREEZE.md`, `WORKLOG.md`

## D121 — 2026-04-20 — accepted

- Decision: future MLXs runtime/performance work must follow the strengthened autonomous integrated-refactor policy in:
  - `AGENTS.md`
  - `docs/execution/REDESIGN_MODE.md`
  - `docs/execution/REFACTOR_POLICY.md`
- Why:
  - the repo needs durable operating rules that prevent drift back into micro-family churn
  - one main front, integrated-refactor-or-close gating, and immediate success/failure continuation must be explicit in repo files rather than implicit in chat
- Evidence:
  - D120
  - `docs/execution/REFACTOR_POLICY.md`

## D122 — 2026-04-20 — accepted

- Decision: worktree / branch hygiene is now a binding part of MLXs execution policy.
- Why:
  - risky redesign paths need isolated worktrees and branches
  - closed clean paths must be cleaned quickly instead of accumulating
  - retained dirty paths must have an explicit recorded reason
- Evidence:
  - `AGENTS.md`
  - `docs/execution/REFACTOR_POLICY.md`
  - current repo worktree inventory after `M01.PC1`

## D123 — 2026-04-20 — accepted

- Decision: remove the clean closed M01 worktrees and branches for `R17`, `R18`, `R19`, `R20`, `R21`, `R28-control`, and `R28-integrated`, and retain the dirty historical M01 worktrees only with an explicit recorded reason.
- Why:
  - those closed clean paths were already reachable from the active baseline and no longer justified their own parked worktrees
  - the retained dirty paths still carry unreviewed local probe/docs state, so autonomous deletion would be unsafe
- Evidence:
  - current git worktree audit from `refactor/core-exec`
  - clean branch ancestry check against `HEAD`

## D124 — 2026-04-20 — accepted

- Decision: `R30` chooses `1. lower-level internal batch redesign` as the next exact path.
- Why:
  - the local no-patch `R30` lower-boundary probe held exact output parity while reproducing the same late-request starvation shape (`129/255` vs `4/131`)
  - below that starvation, attention, mask, and cache-update timings stayed tiny and did not justify a standalone MLX-op or custom-kernel pivot
  - the measured lower-boundary split is instead the owner/eval pipeline:
    - MLXs fast-batch step stayed dominated by `mx.eval`
    - upstream generation step stayed much heavier on `mx.async_eval`
  - that makes the next real lever an internal generation-owner / eval-discipline redesign below the closed prompt-owner family
- Evidence:
  - `/tmp/mlxs_m01_r30_substrate_local_20260420_1.json`
  - `/tmp/mlxs_m01_r30_substrate_local_20260420_2.json`
  - current `src/mlxs/batch/{scheduler.py,fast_batch.py}`
  - current `src/mlxs/runtime_core/batch_progression.py`
  - installed `mlx_lm.generate::{PromptProcessingBatch,GenerationBatch,BatchGenerator}`

## D125 — 2026-04-20 — rejected

- Decision: reject the first integrated `R30` runtime candidate `parallel fast generation owners plus prompt-prefill handoff`.
- Why:
  - local staggered truth-first compare preserved exact parity and fixed the late-request steps to upstream-like `4/131`
  - but the real staggered surface still missed badly:
    - `mlxs/mlx_lm requests_per_s ~0.70730x`
    - `p95 completion ~1.41507x`
  - candidate lower-boundary probe showed the deeper failure mode:
    - MLXs `fast_batch_step` calls doubled to `256`
    - upstream `generation_batch_step` calls stayed `133`
    - the candidate repaired semantics by running parallel generation owners, but that duplicated active generation work instead of extending one owner
- Evidence:
  - `/tmp/mlxs_m01_r30_staggered_local_20260420_2.json`
  - `/tmp/mlxs_m01_r30_staggered_decomp_local_20260420_2.json`
  - `/tmp/mlxs_m01_r30_substrate_candidate_local_20260420_1.json`

## D126 — 2026-04-20 — accepted

- Decision: keep `R30` active and rerank the next exact path to `one active extendable dynamic generation owner on the staggered surface`.
- Why:
  - the rejected first candidate proves that fixing prompt/generation sequencing alone is insufficient if it creates parallel generation owners
  - the next remaining internal lever is a single active owner that can admit late prompt-completed rows without doubling generation stepping
  - this keeps the work internal and architectural; built-in MLX and custom-kernel pivots remain unjustified
- Evidence:
  - D124
  - D125
  - `R25` freeze against paying extendable-cache tax directly on the aligned hot path

## D127 — 2026-04-21 — rejected

- Decision: reject the last acceptable `R30` candidate `one active extendable dynamic generation owner on the staggered surface` and close the `R30` family.
- Why:
  - local truth-first compare preserved exact parity and kept the repaired late-request step positions
  - but the real staggered surface still failed clearly:
    - `mlxs/mlx_lm requests_per_s ~0.72761x`
    - `p95 completion ~1.37504x`
  - this was the last acceptable attempt inside the `R30` family
- Evidence:
  - `/tmp/mlxs_m01_r30_staggered_local_20260421_1.json`
  - `/tmp/mlxs_m01_r30_staggered_decomp_local_20260421_1.json`
  - `/tmp/mlxs_m01_r30_substrate_candidate_local_20260421_1.json`

## D128 — 2026-04-21 — accepted

- Decision: open the next active path as `M01.R31 mixed-offset batch cache substrate redesign on the staggered surface`.
- Why:
  - `R30` proved that generation-owner lifetime alone is no longer the right scope
  - the final candidate fixed the owner semantics but still failed the real surface gate, which points to a broader blocking issue:
    - mixed-offset cache representation
    - mask construction
    - rope / model-boundary behavior
  - this is a broader thesis than `R30`, not another same-family variant
- Evidence:
  - D127
  - `R24` cache-substrate breadth ranking
  - `R25` static-hot-path rejection of extendable cache tax
