# M01.R26-S5 True-Overlap Prompt-Progress Substrate Redesign

Updated: 2026-04-20

## Why this path exists

`R26-S4` kept the right prompt-owner semantics and added dedicated-stream chunked prompt progress while `_SharedFastBatch` stayed hot.

It proved:

- late request first-token step can match upstream exactly
- late request completion step can stay near upstream
- prompt-side semantics are no longer the blocker

It failed because the wall-clock surface still regressed, which means Layer 3 prompt-owner budgeting alone is not enough.

## Main thesis

- `R26-S5 true-overlap prompt-progress substrate redesign`

Core idea:

- preserve the prompt-owner admission semantics proven by `R26-S4`
- move the remaining cost target below the current Layer 3 prompt-owner loop
- redesign the prompt-progress substrate so chunked late-prompt work is cheaper or more truly overlapped than repeated Layer 3-managed model calls

## Fallback thesis

If a lower-layer substrate redesign is still too broad as the next move, first produce a tighter no-patch substrate audit around prompt-progress chunk execution cost and stream interaction so the first `R26-S5` slice is attributable before patching.

## Do not pursue

- no more scheduler-local prompt budgeting retries under the same `R26-S4` shape
- no more attempts to fix this with only admission policy or queue sequencing
- no return to full dynamic owner or synchronous whole-prompt progression

## Result

`R26-S5` is now rejected locally.

Local staggered control/candidate:

- control: `/tmp/mlxs_m01_r26s5_staggered_control_local_20260420_1.json`
- candidate: `/tmp/mlxs_m01_r26s5_staggered_candidate_local_20260420_1.json`

Result:

- requests/s `candidate/control ~1.00040x`
- p50 TTFT `~1.04423x`
- p95 completion `~1.04108x`
- exact output parity: `True`

Candidate decomposition:

- `/tmp/mlxs_m01_r26s5_staggered_decomp_candidate_local_20260420_1.json`
- late first-token step `4`
- late completion step `130`

Interpretation:

- this final descent preserved the corrected semantics but did not create a strong enough throughput lever
- the `R26` batch prompt-progress family is now closed for this phase
