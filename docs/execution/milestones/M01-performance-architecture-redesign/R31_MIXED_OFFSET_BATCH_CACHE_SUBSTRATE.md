# M01.R31 Mixed-Offset Batch Cache Substrate

Updated: 2026-04-21

## Why this path exists

`R30` is now closed for this phase.

What `R30` proved:

- the current accepted MLXs staggered surface is still materially behind
- the surviving bottleneck is not a standalone attention/mask/kernel problem on the accepted baseline
- the first integrated `R30` candidate failed because it duplicated generation work with parallel owners
- the last acceptable `R30` candidate failed even with one owner

Conclusion:

- generation-owner / eval-discipline redesign alone is exhausted for this phase
- the next valid path must be broader than `R30`

## Main thesis

- `R31 mixed-offset batch cache substrate redesign on the staggered surface`

Core idea:

- redesign the dynamic-path-only cache/mask/rope substrate needed for mixed-offset batching
- preserve the aligned static hot path
- enable a future one-owner extendable generation path without paying the current mixed-offset substrate tax

## Fallback thesis

If source archaeology and bounded probes do not isolate a viable substrate-level redesign below the closed `R30` family, close the open AC2 redesign front for this phase and rerank globally instead of forcing another batch descent.

## Do not pursue

- no more `R30` owner/eval variants
- no more prompt-owner budget variants
- no more parallel fast-owner shapes
- no standalone built-in-MLX or custom-kernel pivot on current evidence
