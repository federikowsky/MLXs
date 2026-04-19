# Status

Updated: 2026-04-19

## Active milestone

- `M01-performance-architecture-redesign`

## Active workstream

- `M01.R11 post-R10 acceptance closure rerank`

## Strategic objective

- Move beyond exhausted micro-fix families into a redesign capable of producing a stronger practical win over `mlx_lm`.

## Priority order

1. core dominance first
2. general-path parity/superiority
3. serving/orchestration strength
4. compiled stabilization
5. product-ready standard

## Accepted baseline summary

- Durable git checkpoint committed: `fcb69de` captures the accepted promoted baseline through `R10` plus the `R11` handoff state.
- Sampled-pure short Layer 2 win promoted.
- Compile decode eligibility promoted with prompt-aware gate `<=512`.
- Chat optional `prompt_toolkit` boundary fix promoted.
- Qwen attention/RoPE metadata compatibility fix promoted.
- Qwen long-prompt decode benchmark-surface fix promoted.
- Llama-3.2-3B long-decode benchmark-surface fix promoted.
- Shared Layer 1 + Layer 3 aligned plain-KV batch progression kernel promoted.
- AC2 scheduler-level baseline exists.
- AC2 batch-host restoration promoted as structural cleanup, not throughput lever.
- Serving/live-stack/operator-visibility families materially closed for this phase.

## Acceptance reality

- `AC1` open on broadened accepted set.
- `AC2` open.
- `AC13` blocked-for-now on current accepted baseline.

## Current accepted reference set

- `mlx-community/Llama-3.2-1B-Instruct-4bit`
  - `256 decode 0.99997x`
  - `2048 decode 1.07037x`
- `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
  - `256 decode 1.02031x`
  - `2048 decode 1.02585x`
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
  - `256 decode 1.01699x`
  - `2048 decode 1.01974x`

None clears `AC1 >= 1.10x decode`.

## Current blockers

- Local seam families are mostly exhausted.
- Current accepted wins on Qwen and Llama 3B long decode are benchmark-surface scoped, not architectural.
- AC2 scheduler baseline still shows real throughput deficit (`256 req/s 0.91190x`, `2048 req/s 0.92751x`).
- AC13 best fair prefill tuning stays below bar (`best eager prefill 1.02808x < 1.05x`).
- Deep research result is written down; `run_greedy` lookahead adoption is promoted; remaining blockers are architectural, not informational.
- Next redesign work must choose the correct consumer after `run_greedy` without reopening rejected progression shapes.
- The broad `general_path` prepared-step branch slice was mixed and is now rejected and reverted.
- The narrowed `general_path` enriched top-logprobs-only slice is also rejected and reverted.
- The first-ranked resident completion-batch object probe is regressive and parity-broken.
- The aligned-cohort resident prefill-to-completion handoff slice preserved parity but stayed too weak/mixed locally (`256 req/s ~0.983x`, `2048 req/s ~1.006x` vs accepted main baseline).
- AC2 remains open until the new promoted batch path is rerun on the accepted authoritative MLXs-vs-`mlx_lm` acceptance surface.
- AC1 remains open; the new batch kernel does not by itself close the single-request acceptance gap.

## Accepted remote baseline

- Host: `llm@169.254.225.109`
- Accepted tree: `/tmp/mlxs_layer1_candidate_20260417_1`
- Authoritative interpreter: `/Users/Shared/mlx-cluster-venv/bin/python`

## Immediate next step

- `R10` is promoted.
- Re-rank from the new accepted state:
  - direct AC2 acceptance closure vs `mlx_lm`
  - AC1 vs AC2 priority after the promoted batch kernel
  - whether the next leverage point is more shared progression reuse or acceptance reruns first
- Keep the scheduler-level AC2 probe as neutral promoted tooling.
