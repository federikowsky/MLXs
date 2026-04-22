# M01.R24-S4 Active-Batch State-Lifetime Redesign

Updated: 2026-04-20

## Classification

`R24-S3` classification:

- `3. evidence points to a broader active-batch state-lifetime redesign rather than a narrow overlap fix`

## Candidate status

This file now records a rejected candidate, not the active path.

Rejected candidate name:

- `R24-S4a resident dual-step _SharedFastBatch owner`

Status:

- attempted
- failed local truth-first validation
- reverted
- frozen as a standalone shape

## Main thesis

Redesign `_SharedFastBatch` around a resident dual-step active-batch owner that explicitly carries:

- current prepared step
- next prepared step
- active rows after filtering
- the state transition from emitted current tokens to ready next-step tokens

The goal is to preserve the second-step readiness benefit exposed by `R24-S3` while reducing the step-0 tax that still makes MLXs slower than the comparable upstream path on the open prompt-`256` target.

## Fallback thesis

If the full dual-step owner is too broad as a first move, restrict the redesign to the seed phase only:

- special-case the transition from prefill output into step 0 / step 1 residency
- preserve the later steady-state loop as-is
- use that slice only to remove the step-0 / step-1 asymmetry before widening further

## Do not pursue

- no more materialization-only changes
- no narrow overlap-only seam tweaks
- no prompt-handoff-only batch slice as the main path
- no Layer 3-only resident batch rewrite
- no event-shaping micro-fixes

## First bounded slice

Introduce a resident dual-step owner inside `_SharedFastBatch` that:

1. stores `prepared` and `next_prepared` as first-class active-batch state
2. separates:
   - current token materialization
   - next-step readiness
   - promotion of `next_prepared -> prepared`
3. keeps filtered survivors aligned without re-creating transient next-step state at the same boundary

## Validation gate

Local:

- overlap probe must show step-0 improvement without losing the step-1 readiness advantage
- scheduler guardrail must improve or at least hold prompt `256` without materially regressing prompt `2048`

Remote:

- run scheduler guardrail first
- only run full direct AC2 compare if prompt `256` is positive or very clearly neutral and prompt `2048` stays non-regressive
