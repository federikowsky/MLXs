# M01.R30 AC2 Substrate Rerank

Updated: 2026-04-21

## Why this path exists

`R29` closes the current `AC1` front for this phase.

Current result:

- `AC1` remains below the program bar, but no justified next implementation family remains on current evidence
- `AC2` remains materially behind on the authoritative staggered late-admission surface
- the `R26` prompt-owner / prompt-progress family is already closed for this phase

Under the current redesign rule, the next move is a lower-level batch rerank, not another `AC1` or `R26` variant.

## Main thesis

- `R30 AC2 lower-level substrate and capability rerank on the staggered batch surface`

Core idea:

- isolate the real remaining batch tax below the closed prompt-owner / prompt-progress family
- classify whether the next lever is:
  - lower-level internal batch architecture
  - built-in MLX capability exploitation
  - custom extension / custom Metal kernel feasibility
  - or no substrate path at all

## Fallback thesis

If the no-patch batch substrate probe does not isolate one clear lower-level lever on the authoritative staggered surface, close the current AC2 front for this phase and rerank globally instead of generating more batch variants by inertia.

## Local lower-boundary probe

Artifacts:

- `/tmp/mlxs_m01_r30_substrate_local_20260420_1.json`
- `/tmp/mlxs_m01_r30_substrate_local_20260420_2.json`

Stable readback across both local seeds:

- exact output parity: `True`
- MLXs late request first-token step: `129`
- MLXs late request completion step: `255`
- `mlx_lm` late request first-token step: `4`
- `mlx_lm` late request completion step: `131`

Lower-boundary timing classification:

- MLXs current lower boundary is still dominated by evaluation-boundary timing rather than raw model ops:
  - `fast_batch_step mx.eval_s ~0.892s`
  - `fast_batch_step mx.async_eval_s ~0.144-0.152s`
  - `decode_active mx.eval_s ~1.006-1.008s`
- upstream generation is shifted much more heavily into async next-step ownership:
  - `generation_batch_step mx.async_eval_s ~0.919-0.925s`
  - `generation_batch_step mx.eval_s ~0.346s`
- attention, mask, and cache-update timings stay tiny on both sides relative to that split

Interpretation:

- current accepted MLXs still has the architectural late-request starvation already proven by `R26`
- below that starvation, the surviving measurable lever is an internal generation-owner / eval-discipline problem
- this is not currently a convincing standalone MLX-op, custom-extension, or custom-Metal-kernel bottleneck

## Exact rerank

Chosen next path:

1. `lower-level internal batch redesign`

Rejected as standalone next paths on current evidence:

2. `built-in MLX capability exploitation`
   - streams / `async_eval` remain relevant ingredients, but not as a standalone path
3. `custom extension / custom Metal kernel feasibility path`
   - not justified because the probe did not isolate one dominant op-level bottleneck
4. `close AC2 for this phase and rerank globally`
   - not justified because one clear internal lower-level lever still remains

## Immediate continuation

- `R30` stayed active through its last acceptable family attempt.
- That family is now closed.
- Do not reopen `R26` scheduler-layer prompt-owner variants under a new name.
- Do not pivot to custom-kernel work unless a later probe isolates a true op-level bottleneck.

## First integrated runtime candidate

Candidate shape:

- resident current/next token lifetime inside the fast generation owner
- prompt-prefill handoff into a new fast generation owner
- scheduler orchestration over multiple lower-level fast owners

Local truth-first compare:

- artifact: `/tmp/mlxs_m01_r30_staggered_local_20260420_2.json`
- requests/s `~0.70730x`
- generated_tok/s `~0.70730x`
- p50 TTFT `~0.94736x`
- p95 completion `~1.41507x`
- exact parity: `True`

Local decomposition:

- artifact: `/tmp/mlxs_m01_r30_staggered_decomp_local_20260420_2.json`
- MLXs late first-token/completion steps: `4/131`
- `mlx_lm` late first-token/completion steps: `4/131`

Candidate lower-boundary readback:

- artifact: `/tmp/mlxs_m01_r30_substrate_candidate_local_20260420_1.json`
- MLXs `fast_batch_step` calls: `256`
- upstream `generation_batch_step` calls: `133`
- interpretation:
  - the candidate fixed the ownership sequencing
  - but it did so by duplicating active generation owners rather than extending one owner

Classification:

- rejected

## Next exact path

Chosen continuation:

1. `one active extendable dynamic generation owner on the staggered surface`

Why:

- separate parallel fast owners are now ruled out
- the surviving lever is still internal batch ownership and eval discipline
- the next design must admit late prompt-completed rows without doubling generation stepping
- built-in MLX and custom-kernel pivots remain out on current evidence

## Do not pursue

- no more `R26` scheduler-layer prompt-owner / prompt-progress variants
- no more scheduler-local admission tweaks under a new name
- no automatic custom-kernel pivot without a measured substrate bottleneck
- no more parallel fast generation owners as the primary staggered design

## Last acceptable one-owner candidate

Candidate shape:

- one active extendable dynamic generation owner
- dynamic-only cache substrate
- no parallel fast generation owners

Local truth-first compare:

- artifact: `/tmp/mlxs_m01_r30_staggered_local_20260421_1.json`
- requests/s `~0.72761x`
- generated_tok/s `~0.72761x`
- p50 TTFT `~0.97741x`
- p95 completion `~1.37504x`
- exact parity: `True`

Local decomposition:

- artifact: `/tmp/mlxs_m01_r30_staggered_decomp_local_20260421_1.json`
- MLXs late first-token/completion steps: `4/131`
- `mlx_lm` late first-token/completion steps: `4/131`

Candidate lower-boundary readback:

- artifact: `/tmp/mlxs_m01_r30_substrate_candidate_local_20260421_1.json`
- interpretation:
  - one-owner semantics hold
  - the mixed-offset dynamic path is still too slow to justify promotion

Classification:

- rejected
- `R30` family closed for this phase

## Post-R30 consequence

- The owner/eval family is exhausted.
- The next valid path must be broader:
  - `M01.R31 mixed-offset batch cache substrate redesign on the staggered surface`
- Built-in MLX and custom-kernel pivots remain out until a future probe isolates a true op-level bottleneck.
