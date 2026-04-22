# M01.R26-S4 Prompt-Owner Budgeted Progression Redesign

Updated: 2026-04-20

## Why this path exists

`R26-S3` completed the diagnosis and rejected two concrete runtime shapes:

- `R26-S1 full dynamic late-admission owner`
- `R26-S2 concurrent late prefill plus legacy continuation handoff`

The durable result was:

- the admission semantics from `R26-S2` were directionally correct
- the remaining loss is the cost of doing whole-prompt synchronous work inside the live scheduler step

This file records the next explicit implementation path so the file-backed system does not ambiguously reuse `R26-S3` for a new runtime slice.

## Main thesis

- `R26-S4 integrated prompt-owner budgeted progression redesign`

Core idea:

- keep `_SharedFastBatch` hot and unchanged on the static aligned path
- introduce a real prompt-side owner for late arrivals
- meter prompt work by chunk or budget across scheduler steps instead of paying the full prompt in one step
- preserve the useful admission semantics already proven by `R26-S2`

## Fallback thesis

If the first budgeted owner still misses throughput, keep the new staggered compare and decomposition tooling as the governing surface and rerank the next path around lower-level prompt-prefill/runtime support rather than another scheduler-local continuation tweak.

## Do not pursue

- no more reuse of the `R26-S3` label for runtime implementation work
- no more full dynamic owner retries
- no more synchronous whole-prompt prefill inside one live scheduler step
- no more tiny scheduler seam churn that does not redesign the prompt owner itself

## Result

`R26-S4` is now rejected locally.

Local staggered control/candidate:

- control: `/tmp/mlxs_m01_r26s4_staggered_control_local_20260420_1.json`
- candidate: `/tmp/mlxs_m01_r26s4_staggered_candidate_local_20260420_1.json`

Result:

- requests/s `candidate/control ~0.91001x`
- p50 TTFT `~0.99142x`
- p95 completion `~1.14951x`
- exact output parity: `True`

Candidate decomposition:

- `/tmp/mlxs_m01_r26s4b_staggered_decomp_candidate_local_20260420_1.json`
- late first-token step `4`
- late completion step `130`

Interpretation:

- the prompt-owner semantics are now good enough
- the remaining bottleneck is lower-level prompt-progress substrate cost, not another prompt-owner budget iteration
