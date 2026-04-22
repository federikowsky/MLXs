# M01.R27 Post-R26 Global Rerank

Updated: 2026-04-20

## Why this path exists

The `R26` staggered prompt-progress family is now exhausted for this phase:

- `R26-S1` full dynamic late-admission owner: rejected
- `R26-S2` concurrent late prefill plus legacy continuation handoff: rejected
- `R26-S4` scheduler-layer budgeted prompt owner: rejected
- `R26-S5` lower-layer true-overlap prompt-progress substrate redesign: rejected

The family fixed the late-admission semantics enough to match or nearly match upstream step placement, but it did not produce a strong enough throughput lever on the accepted staggered batch surface.

## Ranked next path

- `R27 global rerank after closing the R26 batch prompt-progress family`

Current global ranking:

1. `AC1` accepted-surface leadership remains the strongest open target
2. repaired direct `AC2` static surface remains near-closed
3. `AC2` staggered prompt-progress family is closed-for-this-phase
4. `AC13` remains blocked-for-now

## Fallback

If new evidence later reopens batching, it must come from a materially different batch family than the closed `R26` prompt-progress family.

## Do not pursue

- no more prompt-owner / prompt-progress batch variants under `R26`
- no more scheduler-layer or substrate overlap retries in the same batch family without new evidence
