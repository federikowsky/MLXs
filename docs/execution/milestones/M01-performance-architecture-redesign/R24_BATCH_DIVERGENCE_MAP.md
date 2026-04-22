# M01.R24 Current Batch Divergence Map

Updated: 2026-04-20

## Phase 0 — Exact framing

### Current accepted AC2 surface

Authoritative current accepted batch surface is the repaired direct comparator, not the stale post-`R10` collapse ledger.

Approved remote direct comparator:

- artifact: `/tmp/mlxs_ac2_direct_compare_candidate_remote_20260419_2.json`
- prompt `256`
  - `mlxs/mlx_lm requests_per_s ~0.99661x`
  - exact output parity: `True`
  - MLXs TTFT slightly better, completion tail near-neutral/slightly worse
- prompt `2048`
  - `mlxs/mlx_lm requests_per_s ~1.09404x`
  - exact output parity: `True`
  - MLXs TTFT and completion are both positive

### What changed because of the repaired direct comparator

Superseded beliefs:

- AC2 is not catastrophically behind on the current accepted baseline
- prompt `2048` is not the dominant AC2 blocker anymore
- old pre-`R10` and stale post-`R10` collapse ledgers are historical only, not the current acceptance ledger

Current truth:

- prompt `256` is the only meaningful open AC2 closure target on the repaired direct surface
- prompt `2048` is already a real positive accepted surface and must be preserved

### Why batch is the strongest open redesign target again

- `R23` closed the adjacent single-request front negatively
- AC1 bounded single-request families are frozen for now
- AC2 remains the strongest adjacent open performance target with a living architectural lever
- the remaining AC2 gap is small enough that stale batch micro-fix ranking is invalid, but still large enough that fresh architecture ranking is justified

## Phase 1 — Current batch divergence map

### Progression ownership

MLXs current code:

- top-level owner is still `BatchScheduler`
- current implementation is split:
  - narrow resident fast path: `_SharedFastBatch`
  - broad legacy scheduler path: `_pending` + `_active` per-sequence state, grouped decode, merge/scatter, inline event shaping

`mlx_lm` current code:

- top-level owner is `BatchGenerator`
- it owns two resident internal owners:
  - `PromptProcessingBatch`
  - `GenerationBatch`

Tax classification:

- architectural tax in MLXs is the split ownership model
- the fast path is real, but still adjunct rather than canonical

### Resident vs rebuilt batch state

MLXs:

- `_SharedFastBatch` keeps resident:
  - `BatchCoreState`
  - current prepared tokens/logits
  - row list
- legacy path rebuilds grouped cache state and keeps per-sequence `current_token`
- fast path cannot act as the universal resident batch owner

`mlx_lm`:

- `PromptProcessingBatch` and `GenerationBatch` are both resident and first-class
- generation state can be extended and filtered in place

Tax classification:

- architectural tax
- current MLXs residency is too narrow and conditional

### Cache ownership

MLXs:

- `BatchCoreState` only supports:
  - `filter_rows()`
  - `extract_row()`
- legacy scheduler still merges/scatters plain `KVCache` state on grouped decode
- no resident prompt-batch cache owner exists

`mlx_lm`:

- prompt/generation batches rely on cache operations that support:
  - merge
  - extend
  - filter
  - extract
  - prepare/finalize for padded prompt processing

Tax classification:

- architectural tax
- current MLXs cache substrate is sufficient for `R10` fast-path promotion, not for canonical continuous residency

### Token / logits / state lifetime

MLXs:

- fast path keeps prepared tokens/logits only
- legacy path keeps per-sequence `current_token` and event list
- no resident `current + next` generation batch owner across the general scheduler surface

`mlx_lm`:

- `GenerationBatch` owns:
  - current tokens
  - next tokens
  - current logprobs
  - next logprobs
  - token contexts
  - matcher states
  - max-token counters

Tax classification:

- architectural tax for extensibility and continuous batching
- only incidental for the narrow greedy direct surface

### Scalarization / materialization timing

MLXs:

- fast path materializes prepared batch tokens, then `_FastBatchRow.append()` decodes text and builds `TokenEvent`
- legacy path also decodes text and builds `TokenEvent` in scheduler hot loops

`mlx_lm`:

- generation batch evaluates current tokens/logprobs, then `next()` emits response objects and filters

Tax classification:

- incidental tax today
- not the main current rerank driver

### Active-set filtering

MLXs:

- `_SharedFastBatch` can filter finished rows
- it cannot extend a live active batch with newly prompt-completed rows
- legacy path fallback is required as soon as the surface stops matching the narrow activation rules

`mlx_lm`:

- generation batch filters in place and can be extended from prompt processing

Tax classification:

- architectural tax

### Prompt processing / prefill-to-decode handoff

MLXs:

- fast path only activates when:
  - batch is empty
  - cohort is aligned
  - request features are minimal
- prompt processing and generation do not share one resident owner across the general surface

`mlx_lm`:

- `PromptProcessingBatch` can process prompts, split finished prompts, and extend `GenerationBatch`
- this resident handoff is canonical, not exceptional

Tax classification:

- architectural tax
- strongest remaining current candidate because:
  - prompt `256` is the only open direct-surface target
  - prompt `2048` already wins, which implies the decode progression core is much healthier than before

### Event shaping location

MLXs:

- scheduler / `_FastBatchRow` still own text decode and `TokenEvent` construction

`mlx_lm`:

- generation batch returns response objects from inside the generation owner

Tax classification:

- incidental today
- do not pursue as the main thesis

### Queue / admission coupling

MLXs:

- `BatchScheduler` couples:
  - admission
  - eligibility checks
  - prompt processing
  - active decode progression selection
  - event shaping

`mlx_lm`:

- `BatchGenerator` still couples queueing at the top level, but prompt and generation owners are internally separated and resident

Tax classification:

- architectural tax
- especially relevant for making the resident path canonical instead of exceptional

### Architectural vs incidental tax summary

Architectural taxes:

1. resident batch ownership is split and conditional
2. prompt processing and generation do not share a canonical resident handoff
3. cache substrate is too narrow for canonical continuous residency
4. active batch cannot be extended in place from newly prompt-completed rows
5. scheduler still couples admission/prompt/decode/event concerns

Incidental taxes:

1. per-row text decode / `TokenEvent` shaping
2. token scalarization details
3. remaining per-step response shaping details

## Phase 2 — Bottleneck ranking from current truth

1. `prompt processing / prefill-to-decode resident handoff`
   - current open target is prompt `256`
   - direct comparator says long-prompt decode is already positive
   - fixed-cost boundary and ownership tax now dominate the rerank

2. `active generation owner cannot extend in place`
   - current resident fast path is batch-empty-only and aligned-cohort-only
   - this keeps the canonical owner too narrow

3. `cache substrate is still too narrow for canonical residency`
   - `filter_rows/extract_row` is not enough for an upstream-like prompt→generation resident pipeline

4. `scheduler-level coupling of admission/prompt/decode/event logic`
   - now secondary to the first three, but still architectural

5. `event/materialization placement`
   - still incidental
   - do not rank as the next main batch bet

## Phase 3 — Redesign thesis ranking

### Ranked thesis

Promote batch residency from special-case adjunct to canonical architecture by introducing a resident `PromptBatch` + extendable `ActiveBatch` pair over shared Layer 1 batch progression, reducing `BatchScheduler` to admission/orchestration plus rich-feature fallback.

Why first:

- it directly attacks the remaining prompt-`256` fixed-cost / handoff tax
- it preserves the `R10` lesson that shared Layer 1 ownership matters
- it avoids repeating the rejected Layer 3-only `R9` family
- it makes the current fast path canonical instead of exceptional

### Fallback thesis

If the full dual-owner redesign is still too broad, first make the resident fast path extendable by introducing a bounded resident prompt-prefill handoff slice that can append prompt-completed rows into an already-active batch without dropping back to per-sequence scheduler state.

Why fallback only:

- it is smaller
- but it still risks becoming another weak one-step handoff if not anchored in a more canonical ownership model

### Do-not-pursue list

- no reopening `R6` resident completion-batch object as a standalone path
- no reopening `R7` one-step handoff as a standalone path
- no Layer 3-only resident batch rewrite
- no merge/scatter/event-shaping micro-fixes as the main thesis
- no single-request helper/front-step families
- no `general_path`
- no AC13

## Phase 4 — Current no-patch result and next active path

### `R24-S1` result

Artifacts:

- local: `/tmp/mlxs_m01_r24_fast_batch_decomp_local_20260420_1.json`
- remote: `/tmp/mlxs_m01_r24_fast_batch_decomp_remote_20260420_1.json`

Stable finding:

- MLXs current fast path activation is not the main remaining gap
- the consistent slower component is the first active-batch generation step

Local summary:

- MLXs activation vs upstream activation: `~0.97728x`
- MLXs first-step vs upstream first-step: `~1.67253x`
- MLXs total vs upstream total: `~1.04600x`

Remote summary:

- MLXs activation vs upstream activation: `~0.93147x`
- MLXs first-step vs upstream first-step: `~2.16977x`
- MLXs total vs upstream total: `~0.95717x`

Interpretation:

- prompt-body plus prompt→generation handoff is already competitive
- the first active-batch generation step is the stable slower component
- the first bounded batch slice should therefore be chosen from inside the active-batch owner, not from prompt-batch handoff alone

### `R24-S2` result

Artifacts:

- local first-step control:
  - `/tmp/mlxs_m01_r24_fast_batch_first_step_local_20260420_1.json`
- local first-step candidate:
  - `/tmp/mlxs_m01_r24_fast_batch_first_step_candidate_local_20260420_1.json`
- remote first-step control:
  - `/tmp/mlxs_m01_r24_fast_batch_first_step_remote_20260420_1.json`
- remote first-step candidate:
  - `/tmp/mlxs_m01_r24_fast_batch_first_step_candidate_remote_20260420_2.json`
- local scheduler control:
  - `/tmp/mlxs_m01_r24_control_local_scheduler_20260420_1.json`
- local scheduler candidate:
  - `/tmp/mlxs_m01_r24_candidate_local_scheduler_20260420_1.json`
- remote scheduler control:
  - `/tmp/mlxs_m01_r24_control_remote_scheduler_20260420_1.json`
- remote scheduler candidate:
  - `/tmp/mlxs_m01_r24_candidate_remote_scheduler_20260420_1.json`

Stable findings:

- isolated first-step materialization is a real micro-tax
- but removing grouped current+next token eval is not a promotable slice on the real AC2 target

Evidence:

- remote first-step micro surface:
  - `schedule_next_s` stayed better than comparable upstream step
  - `materialize_s` dominated current MLXs first-step cost
  - candidate/control first-step total `~0.45550x`
  - candidate/control `materialize_s ~0.14887x`
- remote scheduler guardrail:
  - prompt `256 candidate/control requests_per_s ~0.95880x`
  - prompt `256` TTFT `~1.07698x`
  - prompt `256` p95 completion `~1.04297x`
  - prompt `2048 candidate/control requests_per_s ~1.04748x`

Interpretation:

- current grouped materialization is carrying useful next-step readiness / overlap
- one-step materialization surgery improves the micro probe but regresses the real open target
- the next evidence slice must measure overlap across consecutive active-batch steps

Classification:

- rejected as non-promotable

### Next active path

- `M01.R24-S3 active-batch overlap decomposition`

Purpose:

- explain why the materialization-only candidate improves the isolated first step but regresses the real prompt-`256` scheduler surface
- decompose at least two consecutive active-batch steps to isolate:
  - current-token materialization
  - next-token readiness / overlap
  - second-step start latency
  - whether grouped eval is masking a downstream scheduler stall

Immediate deliverable:

- a no-patch local probe across consecutive active-batch steps
- this is missing evidence, not a stop condition

If the probe confirms overlap/readiness is the true dependency, the first bounded implementation slice should be:

- active-batch owner overlap redesign over the existing shared Layer 1 batch progression substrate

### `R24-S3` result

Artifacts:

- local overlap probe:
  - `/tmp/mlxs_m01_r24_fast_batch_overlap_local_20260420_1.json`
- remote overlap probe:
  - `/tmp/mlxs_m01_r24_fast_batch_overlap_remote_20260420_1.json`

Local summary:

- MLXs step 0 total `~0.08893s` vs upstream step 0 total/core `~0.06335s`
- MLXs step 1 total `~0.02028s` vs upstream step 1 total/core `~0.02411s`

Remote summary:

- MLXs step 0 total `~0.02561s` vs upstream step 0 total/core `~0.01186s`
- MLXs step 1 total `~0.01451s` vs upstream step 1 total/core `~0.01407s`
- MLXs step 1 schedule alone `~0.00194s` vs upstream step 1 core `~0.01405s`

Interpretation:

- overlap/readiness is real
- but it is not an isolated narrow seam
- the stronger reading is that active-batch current/next prepared state lifetime is the real owner problem

## Final classification

- classification: `3. evidence points to a broader active-batch state-lifetime redesign rather than a narrow overlap fix`
- next active path: `M01.R24-S4 active-batch state-lifetime redesign`

See:

- `docs/execution/milestones/M01-performance-architecture-redesign/R24_ACTIVE_BATCH_STATE_LIFETIME.md`
