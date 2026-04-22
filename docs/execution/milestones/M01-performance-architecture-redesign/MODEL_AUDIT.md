# M01 Model Audit

Updated: 2026-04-20

## Purpose

Focused model-correctness sanity checks for the benchmark-relevant `Llama` and `Qwen` families used in current `AC1`/guardrail work.

These are current-target audits, not exhaustive proofs for every variant in each family.

## Llama

Target audited:
- `mlx-community/Llama-3.2-1B-Instruct-4bit`

Checks completed:
- config interpretation parity against `mlx_lm`
- attention metadata parity:
  - heads
  - kv-heads
  - head-dim
- first-layer projection shape parity
- controlled prompt parity:
  - prefill token parity
  - prefill logits max-abs diff `0.0`
  - first decode logits max-abs diff `0.0`
  - cache offsets match
- first transformer block output parity:
  - layer-0 max-abs diff `0.0`

Classification:
- `implementation looks correct enough; benchmark deficit is likely runtime/architecture`

## Qwen

Target audited:
- `mlx-community/Qwen2.5-1.5B-Instruct-4bit`

Checks completed:
- text-mode wrapper resolution sanity:
  - heads `12`
  - kv-heads `2`
  - head-dim `128`
- controlled prompt parity:
  - prefill token parity
  - prefill logits max-abs diff `0.0`
  - first decode logits max-abs diff `0.0`
  - cache offsets match

Classification:
- `implementation looks correct enough; benchmark deficit is likely runtime/architecture`

## Current implication

For the present `AC1` weakest-case analysis:
- current `Llama-3.2-1B 256` deficit should not currently be treated as primarily a wrapper/config/attention mismatch
- current `Qwen2.5-1.5B 256` guardrail behavior should not currently be treated as primarily a wrapper/config/attention mismatch
- current evidence points more strongly to helper-surface runtime/hot-loop cost
