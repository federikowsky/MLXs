# M01 Tasks

## Active

- [x] Create `RESEARCH.md`.
- [x] Create `ARCHITECTURE_NOTES.md`.
- [x] Complete Phase 0 framing in `RESEARCH.md`.
- [x] Complete Phase 1 source archaeology in `RESEARCH.md`.
- [x] Complete Phase 2 architecture maps in `ARCHITECTURE_NOTES.md`.
- [x] Complete Phase 3 divergence map in `RESEARCH.md`.
- [x] Complete Phase 4 bottleneck model in `RESEARCH.md`.
- [x] Complete Phase 5 ranked redesign thesis + fallback + first bounded slice.
- [x] Complete `M01.R1b` MLX capability inventory addendum.

## Historical completed implementation work

- [x] Initialize `docs/execution/` file-backed operating system.
- [x] Set `M01-performance-architecture-redesign` as active milestone.
- [x] Rank redesign fronts from accepted AC1/AC2/AC13 evidence.
- [x] Perform first architecture archaeology on shared decode/batch hot-path ownership.
- [x] Choose the first redesign family: `resident decode state machine / hot-path unification`.
- [x] Define the exact resident decode progression boundary in Layer 1.
- [x] Decide that the first redesign implementation needs a dedicated worktree.
- [x] Open `MLXs-m01-resident-decode-state-machine`.
- [x] Reject class-based resident progression slice.
- [x] Reject data-only resident progression state slice.
- [x] Reject broad inline helper migration slice.
- [x] Promote lookahead-branch-only helper migration slice.

## Next

- [x] Update `STATUS`, `DECISIONS`, `WORKLOG`, `RISKS`, and `FREEZE` as the research stabilizes.
- [x] Resume implementation only after `M01.R1` closes.
- [x] Begin `M01.R2 runtime_core.run_greedy lookahead-path adoption`.
- [x] Implement generic short-prompt lookahead in `run_greedy`.
- [x] Validate `run_greedy` locally against accepted-sensitive cases.
- [x] Run remote authoritative `run_greedy` probe.
- [x] Rank the next adjacent redesign consumer/path after `run_greedy`.
- [x] Define the smallest bounded `general_path single_request` adoption slice.
- [x] Implement the bounded `general_path` slice in the redesign worktree.
- [x] Reject the broad `general_path` slice after local truth-first validation.
- [x] Rank the enriched top-logprobs-only subpath as the next `general_path` slice.
- [x] Define the exact bounded enriched `top_logprobs`-only slice.
- [x] Reject the narrowed `general_path` `top_logprobs`-only slice after local truth-first validation.
- [x] Begin the next adjacent redesign path after `general_path` exhaustion.
- [x] Complete R5 framing from accepted AC2 artifacts and code.
- [x] Complete MLXs vs `mlx_lm` batch divergence map.
- [x] Rank at most 2 resident batch architecture candidates.
- [x] Choose the first bounded batch architecture slice.
- [x] Begin `M01.R6 resident active completion-batch object slice definition`.
- [x] Define the exact persistent state shape.
- [x] Define the first bounded batch slice boundary.
- [x] Run a no-patch truth-first probe for the resident completion-batch idea.
- [x] Reject `R6` after local truth-first probe.
- [x] Begin `M01.R7 prefill/prompt-processing unification around a resident batch object ranking`.
- [x] Define the exact resident prefill-to-completion handoff slice.
- [x] Implement the aligned-cohort resident handoff slice in the redesign worktree.
- [x] Validate the aligned-cohort resident handoff slice locally on the canonical AC2 scheduler surface.
- [x] Reject `M01.R7` after local AC2 truth-first compare.
- [x] Rank the broader integrated batch-contract redesign after both narrow batch-side slices fail.
- [x] Implement the private resident fast-batch runtime for the canonical AC2 fast path.
- [x] Refactor `BatchScheduler` into orchestration over resident fast batch + accepted legacy fallback.
- [x] Add reusable scheduler-level AC2 probe under `benchmarks/mlxs_vs_mlx_lm`.
- [x] Validate `R9` locally on focused scheduler/batched-serving suites and the canonical AC2 compare.
- [x] Run remote-authoritative `R9` AC2 compare against a fresh remote mirror of the accepted local baseline.
- [x] Reject `R9` after mixed remote result.
- [x] Rank the broader shared Layer 1 + Layer 3 batch-capable progression redesign.
- [x] Implement the aligned plain-KV shared batch progression kernel.
- [x] Refactor `BatchScheduler` to orchestrate shared Layer 1 batch progression plus accepted legacy fallback.
- [x] Carry the scheduler-level AC2 probe into the R10 worktree and validate `R10` locally.
- [x] Run remote-authoritative `R10` AC2 compare on fresh remote mirrors.
- [x] Promote `R10`.
- [x] Create a durable git checkpoint for the accepted promoted baseline through `R10`.
- [ ] Re-rank AC1/AC2/AC13 from the promoted R10 baseline.
