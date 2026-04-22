# M01.R24 Batch-First Redesign Rerank

Updated: 2026-04-20

## Objective

Restart the batch-first redesign front from the repaired current accepted state after `R23` closed the adjacent single-request first-decode family negatively.

## Governing evidence

- Current accepted AC2 direct comparator:
  - prompt `256 requests/s ~0.99661x`
  - prompt `2048 requests/s ~1.09404x`
- `R9` proved that Layer 3-only resident batch redesign can win hard on short prompts and still fail badly on long prompts.
- `R10` proved that shared Layer 1 + Layer 3 progression ownership can improve AC2 and is promotable.
- `R23` now closes the neighboring single-request first-decode front:
  - current MLXs prompt-tail / first-decode boundary is already better than the comparable `mlx_lm` boundary in total wall time
  - the first no-patch integrated boundary variant is materially worse both locally and remotely

## Main thesis

The next high-leverage path is a batch-first resident progression redesign that targets prompt-`256` closure without sacrificing the promoted prompt-`2048` gain.

## Do not pursue

- no reopening `R6` or `R7` as standalone narrow entry slices
- no return to Layer 3-only batch ownership
- no reopening single-request helper/front-step variants as the main front
- no AC13 reopening on this path

## Immediate next step

Build a fresh batch divergence map against the repaired direct comparator and isolate the first bounded redesign target across:

1. prompt processing / prefill ownership
2. active completion state residency
3. row-level cache ownership and filtering
4. event/materialization placement

The first bounded slice must preserve the current prompt-`2048` gain while targeting the remaining prompt-`256` weakness.
