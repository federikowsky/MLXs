# M01 Plan — Performance Architecture Redesign

## Goal

Replace exhausted micro-fix loops with structural runtime redesign that can materially improve AC1 and AC2 versus `mlx_lm`.

## Current mode

- `M01.R11 post-R10 acceptance closure rerank`
- `R10` is promoted.
- Next work reranks AC1/AC2/AC13 from the new accepted baseline rather than implementing another redesign slice immediately.

## Entry evidence

- AC1 remains open on broadened accepted set even after Qwen and Llama 3B benchmark-surface wins.
- AC2 scheduler baseline is authoritative and still behind on throughput.
- AC13 is blocked-for-now on the current accepted baseline.
- Many adjacent seam families were already tested and rejected.

## Redesign hypothesis

The remaining acceptance gaps are now dominated by shared structural hot-path issues:

- resident decode state ownership is still too fragmented
- benchmark fast-path wins are not yet represented as a unified runtime contract
- batch progression still lacks a resident, low-overhead advancement model
- layer boundaries are correct enough to preserve, but the runtime/batch hot path is still too piecemeal

## Proposed fronts

1. `F1 resident decode state machine / hot-path unification`
   - unify prepared-step/lookahead and long decode progression under a stronger Layer 1 contract
   - reduce duplicated benchmark-local vs runtime-local advancement logic
2. `F2 resident batch progression / AC2 throughput architecture`
   - redesign batch progression around resident decode state rather than per-step reconstruction
3. `F3 general-path hot-path unification`
   - narrow general-path enrichment over the redesigned core so richer paths do not reopen hot-path cost
4. `F4 acceptance closure sequencing`
   - re-run AC1/AC2/AC13 only after redesign fronts produce credible structural wins

## Chosen entry front

- `F1 resident decode state machine / hot-path unification`
- Reason:
  - accepted wins on Qwen and Llama 3B long decode currently live in the benchmark helper
  - `runtime_core.run_greedy()` still advances through the simpler `decode_step` loop
  - `general_path` and `batch.scheduler` each own separate decode progression logic above or outside Layer 1
- Immediate consequence:
  - redesign must first define a resident Layer 1 progression contract that can be consumed by both single-request and batch paths without widening Layer 1 outputs

## Target boundary

The first redesign target is a resident Layer 1 progression object/state machine that owns:

- `CoreState` / mutable cache ownership
- prompt/generation counters
- current prepared token/logits handle
- optional next-step prepared/logits handle
- bound execution and termination policy
- bound model step function

It must not own:

- `TokenEvent`
- tokenizer text decode
- stop-sequence text matching
- top-logprobs shaping
- batch admission or queue order

## Initial sequencing

1. rank fronts from accepted evidence
2. map F1/F2 shared state and control-flow ownership
3. define the resident decode progression boundary
4. choose first redesign worktree only after evidence gate
5. keep F3/F4 downstream

## Research sequencing

1. framing
2. source archaeology
3. architecture maps
4. divergence map
5. bottleneck model
6. ranked redesign thesis
7. first bounded implementation slice

## Non-goals

- no new product-surface redesign
- no speculative benchmark breadth expansion before redesign evidence demands it
- no reopening frozen micro-fix seams without new evidence
