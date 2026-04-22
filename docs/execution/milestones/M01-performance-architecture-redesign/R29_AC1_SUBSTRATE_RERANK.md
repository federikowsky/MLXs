# M01.R29 AC1 Substrate Rerank

Updated: 2026-04-20

## Why this path exists

`R28` cleanly removed benchmark-local progression duplication and still failed on the real accepted-sensitive `AC1` surfaces.

Current result:

- `AC1` remains the strongest open target
- helper/core convergence micro-families are already exhausted
- the batch prompt-progress family is closed for this phase
- another contract-level ownership rewrite is not justified by default

Under the current redesign rule, the next move is a lower-level rerank, not another AC1 micro-slice.

## Main thesis

- `R29 AC1 lower-level substrate and capability rerank`

Core idea:

- isolate the real remaining AC1 tax below the rejected progression contract seam
- classify whether the next lever is:
  - lower-level internal architecture
  - built-in MLX capability exploitation
  - custom extension / custom Metal kernel feasibility
  - or no substrate path at all

## Fallback thesis

If the no-patch probe does not isolate one clear substrate-level lever across the accepted-sensitive surfaces, close the AC1 front for this phase and rerank globally instead of generating more AC1 variants.

## Do not pursue

- no more helper/core convergence micro-slices under `R17` / `R18` / `R19`
- no more helper-only token-choreography redesigns under `R22`
- no more benchmark-local progression rewrites after `R28`
- no automatic custom-kernel pivot without a measured substrate bottleneck

## Result

- Main artifact:
  - `/tmp/mlxs_m01_r29_substrate_local_20260420_1.json`
- Local accepted-sensitive readback:
  - exact parity on all local cases
  - decode ratios:
    - `Llama-3.2-1B 256 ~1.01389x`
    - `Llama-3.2-1B 2048 ~1.10801x`
    - `Qwen2.5-1.5B 256 ~1.02524x`
    - `Qwen2.5-1.5B 2048 ~0.97067x`
- Lower-boundary classification:
  - decode `model_forward_s`, attention, cache update/fetch, and mask costs are mixed across cases
  - no single lower-level internal or substrate-level component dominates strongly enough to justify a new `AC1` implementation family
  - built-in MLX capability and custom-kernel paths are not justified for `AC1` on current evidence
- Outcome:
  - close `AC1` for this phase
  - rerank globally to `M01.R30 AC2 lower-level substrate and capability rerank on the staggered batch surface`
