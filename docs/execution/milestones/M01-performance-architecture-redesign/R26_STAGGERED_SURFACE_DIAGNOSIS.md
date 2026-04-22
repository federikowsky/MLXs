# M01.R26 Staggered Surface Diagnosis

Updated: 2026-04-20

## Purpose

Record the first authoritative readback of the staggered late-admission surface and the two rejected `R26` implementation shapes.

## Authoritative current baseline

Remote baseline artifact:

- `/tmp/mlxs_m01_r26_staggered_remote_20260420_1.json`

Prompt `256`, initial `2`, late `1`, late-after-step `1`:

- `mlxs/mlx_lm requests_per_s ~0.85088x`
- `mlxs/mlx_lm generated_tok_per_s ~0.85088x`
- `mlxs/mlx_lm p50 TTFT ~0.99697x`
- `mlxs/mlx_lm p95 completion ~1.12976x`
- exact output parity: `True`

## Decisive decomposition

Local decomposition:

- `/tmp/mlxs_m01_r26_staggered_decomp_local_20260420_1.json`

Remote decomposition:

- `/tmp/mlxs_m01_r26_staggered_decomp_remote_20260420_1.json`

Shared diagnosis:

- initial requests are not the main problem
  - MLXs initial first-token step: `1`
  - `mlx_lm` initial first-token step: `1`
- late request starvation is the decisive current tax
  - MLXs late first-token step: `129`
  - `mlx_lm` late first-token step: `4`
  - MLXs late completion step: `255`
  - `mlx_lm` late completion step: `131`

Conclusion:

- the current accepted MLXs design does not admit late arrivals into active work in a competitive way
- the dominant staggered-surface problem is still owner policy, but it is specifically prompt-side starvation before continuation

## Rejected `R26` implementation shapes

### `R26-S1` full dynamic late-admission owner

Local staggered control/candidate:

- control: `/tmp/mlxs_m01_r26_staggered_control_local_20260420_1.json`
- candidate: `/tmp/mlxs_m01_r26_staggered_candidate_local_20260420_1.json`

Result:

- requests/s `candidate/control ~0.77448x`
- p50 TTFT `~1.02358x`
- p95 completion `~1.34605x`
- exact output parity: `False`

Classification:

- rejected

### `R26-S2` concurrent late prefill plus legacy continuation handoff

Local staggered control/candidate:

- control: `/tmp/mlxs_m01_r26b_staggered_control_local_20260420_1.json`
- candidate: `/tmp/mlxs_m01_r26b_staggered_candidate_local_20260420_1.json`

Result:

- requests/s `candidate/control ~0.99397x`
- p50 TTFT `~0.94485x`
- p95 completion `~1.04957x`
- exact output parity: `True`

Local candidate decomposition:

- `/tmp/mlxs_m01_r26b_staggered_decomp_candidate_local_20260420_1.json`

What it proved:

- late request first-token step moved `129 -> 2`
- late request completion step moved `255 -> 128`
- the owner semantics are directionally correct
- but synchronous late prompt processing inside the main scheduler step adds enough wall-time overhead to miss the throughput gate

Classification:

- rejected as neutral-to-regressive

## Next thesis

Rank next:

- `R27 global rerank after closing the R26 batch prompt-progress family`

Core idea:

- keep `_SharedFastBatch` hot and unchanged
- introduce a real prompt-side owner with bounded chunked progress while generation is live
- avoid synchronous whole-prompt prefill inside the main generation step
- do not force extendable cache or full dynamic generation ownership into the static hot path as the first move
