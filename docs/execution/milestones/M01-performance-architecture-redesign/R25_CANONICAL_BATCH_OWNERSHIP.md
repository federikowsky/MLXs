# M01.R25 Canonical Batch Ownership Redesign

Updated: 2026-04-20

## Why this path exists

`R24-S4` already attempted one concrete candidate inside the broader active-batch state-lifetime family:

- resident dual-step `_SharedFastBatch` owner
- explicit `prepared` + `next_prepared` lifetime inside the existing narrow fast path
- preserve readiness after filtering without changing the outer owner boundary

That candidate failed local truth-first validation and was reverted.

This file defines a genuinely different next path.

## Rejected candidate

Rejected candidate name:

- `R24-S4a resident dual-step _SharedFastBatch owner`

Why it failed:

- it kept the existing narrow aligned-cohort fast path as the canonical owner
- it improved neither the fast-path activation nor the overlap totals locally
- it made the current fast-path activation and overlap probes materially worse

## New thesis

New active thesis:

- `canonical prompt-batch + extendable active-batch ownership redesign`

Core idea:

- stop treating `_SharedFastBatch` as a narrow adjunct owner
- introduce first-class resident batch owners for:
  - prompt processing
  - active generation
- allow active generation state to be extended in place from prompt completion
- move current/next prepared lifetime into that canonical active owner rather than mutating the old fast-path helper in place

## Why it is materially different

This is not another retry of the rejected `R24-S4a` shape.

Rejected `R24-S4a`:

- changed state lifetime inside the existing narrow `_SharedFastBatch`
- preserved batch-empty-only / aligned-cohort-only ownership
- did not change the canonical owner boundary

New `R25` thesis:

- changes the owner boundary itself
- promotes prompt batch and active batch to first-class resident owners
- targets extendability and canonical residency, not just current/next lifetime inside the old fast path
- is aligned with the original `R24` architectural thesis rather than the failed first candidate shape

## First bounded slice

First bounded slice for `R25`:

- create a resident `ActiveBatch` owner contract over shared Layer 1 batch progression that can:
  - hold current prepared state
  - hold next prepared state
  - filter rows
  - accept newly prompt-completed rows through an explicit extend boundary

Keep out of scope for this first slice:

- general queue/admission redesign
- rich feature families outside the canonical AC2 fast path
- `general_path`
- AC13

## Result

`R25` is now rejected on the current canonical static prompt-`256` surface.

Local truth-first result:

- token-resident extendable owner:
  - requests/s `candidate/control ~0.79339x`
  - p50 TTFT `~1.06247x`
  - p95 completion `~1.31978x`
- dual-ready extendable owner:
  - requests/s `candidate/control ~0.73350x`
  - p50 TTFT `~1.02370x`
  - p95 completion `~1.36299x`

Interpretation:

- the static aligned canonical fast path is the wrong place to pay the extendable-cache tax
- the current direct AC2 canonical compare does not exercise the intended late-admission leverage
- the next valid move is a rerank that separates static aligned fast-path authority from dynamic extension authority
