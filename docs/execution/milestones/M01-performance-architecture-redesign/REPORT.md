# M01 Report

Milestone not closed.

## Durable checkpoint

- Commit: `fcb69de`
- Type: milestone checkpoint commit
- Captures:
  - accepted promoted baseline through `R10`
  - promoted shared Layer 1 + Layer 3 aligned plain-KV batch progression kernel
  - promoted scheduler-level AC2 probe tooling
  - file-backed execution state through active handoff `R11`

## Current promoted fronts

- `M01.5` benchmark-helper lookahead branch
- `M01.R2` `runtime_core.run_greedy` lookahead adoption
- `M01.R10` shared Layer 1 + Layer 3 aligned plain-KV batch progression kernel
- `M01.R14` AC2 direct comparator/tooling repair

## Rejected redesign families

- class-based resident progression container
- data-only resident progression container
- broad inline helper migration
- broad and narrowed `general_path` helper migrations
- `R6` resident completion-batch object
- `R7` aligned cohort prefill-to-completion handoff
- `R9` private Layer 3-only resident fast-batch contract
- `R17` run_greedy-only short-prompt materialization contract alignment
- `R18` shared generator-style short-prompt helper/core convergence slice
- `R19` low-overhead inline short-prompt helper/core convergence slice
- `R22` resident token-lookahead helper-contract redesign

## Carry-forward

- milestone remains open
- `R15` result:
  - `AC1` is now the strongest open target
  - `AC2` is near-closed on the repaired direct surface
  - `AC13` remains blocked
- `R16` result:
  - short-prompt helper/core divergence is still material
  - long-prompt `Llama-3.2-1B 2048` no longer points to the same split
- `R17` result:
  - local `run_greedy`-only signal was positive
  - candidate is rejected as non-promotable because it does not cleanly move the current accepted AC1 helper surface
- `R18` result:
  - shared helper/core convergence stayed directionally plausible
  - candidate is rejected because the shared per-token generator shape regressed the weakest accepted short-prompt case locally
- `R19` result:
  - local evidence was strong
  - candidate is rejected because the decisive authoritative weakest-case remote surface still did not improve
- `R20` result:
  - weakest accepted surface is confirmed as `Llama-3.2-1B 256`
  - helper-path `schedule_next` / forward-build cost is the dominant measured bottleneck
  - current `Llama` and `Qwen` model correctness looks clean enough for this target
- `R22` result:
  - local control/candidate signal stayed too small to matter on the weakest case
  - authoritative remote control/candidate rerun stayed slightly regressive in absolute MLXs decode throughput on both `256` and `2048`
  - simplifying helper token choreography is not enough; the next leverage moves below the helper boundary
- `R23` current state:
  - local + remote substrate probes are complete
  - current MLXs total prompt-tail / first-decode boundary is already better than comparable `mlx_lm`
  - the first integrated no-patch boundary probe is materially worse, so no bounded single-request slice is justified
- `R24` current state:
  - current batch divergence map is rebuilt from repaired direct evidence
  - main thesis is now resident prompt-batch + extendable active-batch ownership over shared Layer 1 batch progression
  - `R24-S1` is complete and moved the first bounded slice to the active-batch generation step
  - `R24-S2` is rejected; materialization-only change improved the micro-surface but regressed the real prompt-`256` scheduler target
  - `R24-S3` is complete and classifies the issue as broader active-batch state lifetime rather than a narrow overlap seam
  - `R24-S4a` first concrete candidate is rejected and frozen
  - the broader batch family remains open under a new path, not under recycled `R24-S4`
- `R25` current state:
  - isolated owner-boundary worktree opened and used for truth-first local validation
  - token-resident extendable owner is rejected locally
  - dual-ready extendable owner is rejected locally
  - new staggered late-admission compare proves the dynamic extension gap is much larger than the static canonical AC2 compare reveals
- `R26` current state:
  - authoritative remote staggered baseline is established
  - authoritative remote decomposition shows late-request starvation is the decisive tax
  - `R26-S1` full dynamic late-admission owner is rejected locally
  - `R26-S2` concurrent late prefill plus legacy continuation handoff is rejected locally after fixing the step-level semantics but missing the wall-clock gate
- `R26-S4` current state:
  - budgeted prompt-owner semantics implemented and validated locally
  - late request step positions now match or nearly match upstream
  - wall-clock staggered surface still misses the gate
- `R26-S5` current state:
  - lower-layer prompt-progress substrate redesign implemented and validated locally
  - corrected step positions are preserved
  - wall-clock staggered surface still misses the gate
  - the `R26` family is now closed for this phase
- `R27` current state:
  - closure check is complete
  - `AC1` remains strongest
  - next move is integrated refactor, not another bounded variant
- `R28` current state:
  - integrated canonical progression contract refactor rejected locally
  - benchmark-local progression duplication is no longer treated as the decisive remaining AC1 lever
- `R29` current state:
  - no-patch accepted-sensitive local substrate rerank completed
  - exact parity held on all local cases
  - no single lower-level internal or substrate-level component dominated strongly enough to justify another AC1 family
  - built-in MLX capability and custom-kernel paths are not justified for AC1 from current evidence
  - the current AC1 front is now closed for this phase
- `M01.PC1` current state:
  - repo-level process correction applied
  - the old process is recorded as too slice-fragmented
  - future runtime/performance work must follow the binding autonomous integrated-refactor policy
- `M01.PC1` hygiene consequence:
  - risky redesign paths now require dedicated worktrees / branches by default
  - stale clean M01 worktrees and branches are now cleaned immediately instead of remaining parked
- `R30` current state:
  - local lower-boundary probe completed with exact output parity on two seeds
  - current accepted MLXs still starves the late request (`129/255` steps vs `mlx_lm 4/131`)
  - below that, the measured lower-boundary split is eval-discipline / owner-pipeline cost, not attention/mask/cache-update kernels
  - standalone built-in-MLX and custom-kernel pivots are not justified on current evidence
  - first integrated runtime candidate repaired the late-request step positions but was rejected locally because it duplicated generation stepping
  - the last acceptable one-owner dynamic candidate was also rejected locally
  - `R30` is now closed for this phase
- exact next path:
  - `M01.R31 mixed-offset batch cache substrate redesign on the staggered surface`
