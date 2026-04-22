# M01.R26 Dual-Owner Batch Rerank

Updated: 2026-04-20

## Why this path exists

`R25` tested the current thesis directly in an isolated worktree:

- worktree: `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r25-canonical-batch-ownership`
- candidate family:
  - per-row-offset batch-cache substrate on the canonical fast path
  - token-resident extendable `ActiveBatch` owner
  - dual-ready extendable `ActiveBatch` owner

Both local truth-first candidates regressed the canonical prompt-`256` scheduler surface enough to reject the path as a promotable redesign on that surface.

## Governing evidence

- Current accepted direct AC2 canonical compare remains near-closed but static:
  - prompt `256 requests/s ~0.99661x`
  - prompt `2048 requests/s ~1.09404x`
- `R25` token-resident candidate local scheduler control/candidate:
  - requests/s `~0.79339x`
  - p50 TTFT `~1.06247x`
  - p95 completion `~1.31978x`
  - artifact pair:
    - `/tmp/mlxs_m01_r25_scheduler_control_local_20260420_1.json`
    - `/tmp/mlxs_m01_r25_scheduler_candidate_local_20260420_1.json`
- `R25` dual-ready candidate local scheduler control/candidate:
  - requests/s `~0.73350x`
  - p50 TTFT `~1.02370x`
  - p95 completion `~1.36299x`
  - artifact pair:
    - `/tmp/mlxs_m01_r25_scheduler_control_local_20260420_1.json`
    - `/tmp/mlxs_m01_r25_scheduler_candidate_local_20260420_2.json`
- New staggered local direct compare now measures the late-admission surface that the canonical compare does not:
  - artifact: `/tmp/mlxs_m01_r25_staggered_local_20260420_1.json`
  - prompt `256`
    - `mlxs/mlx_lm requests_per_s ~0.69428x`
    - `mlxs/mlx_lm p50 TTFT ~1.08180x`
    - `mlxs/mlx_lm p95 completion ~1.37731x`
    - exact output parity: `True`

## Ranked thesis

- keep the aligned plain-KV fast batch as the hot canonical owner for the current static AC2 benchmark surface
- treat dynamic late-admission / extendable batching as a separate owner path with its own benchmark authority
- do not force per-row-offset extendability into the aligned hot path unless a future result proves that the aligned surface stays non-regressive

## Fallback thesis

If a separate dynamic owner is still too broad, first expand neutral benchmark tooling around staggered prompt admission and continuous batching so future redesigns are ranked against a surface that actually exercises the intended leverage.

## Do not pursue

- no more attempts to make the canonical aligned prompt-`256` fast path itself carry per-row-offset extendable batch cache state as the first move
- no more token-resident or dual-ready active-owner retries on that canonical static surface
- no more claims that the repaired direct AC2 canonical compare is sufficient authority for late-admission redesign ranking by itself
