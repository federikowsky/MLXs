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
