# Appendix A. Experimental Setup

> **Archival note:** This appendix belongs to the Adaptive KV **V1** / `FULL`-`COMPRESSED`-`EVICTED` evaluation-era report. It is historical source material. For the **frozen** exact adaptive execution-pack baseline and archived track status, see [`../adaptive_kv_turboquant_first.spec.md`](../adaptive_kv_turboquant_first.spec.md) and [`../adaptive_kv_execution_pack_track_archival.md`](../adaptive_kv_execution_pack_track_archival.md).

## A.1 Retained Scope Evaluated

All reported results are within the retained Adaptive KV V1 scope:

| Dimension | Retained scope |
|---|---|
| Generation mode | Single-request generation only |
| Principal evaluation model | `model_type == "llama"` (tables in this report) |
| Principal evaluation baseline | Full-attention Llama with homogeneous `list[KVCache]` |
| Additional retained implementations | Standard `ministral3` on Family B and standard text-only `qwen3_5` on Family C when compatibility passes (not substituting for the Llama tables here) |
| Historical adaptive tiers in this report package | `FULL`, `COMPRESSED`, `EVICTED` |
| Recovery model | Real replay-backed recovery from authoritative source tokens |
| Observer semantics | Observer-only; no query/key/value/logit modification |
| Excluded features | `compile_decode`, legacy `quantized_kv_start`, external cache reuse, unsupported families |

The current retained implementation baseline is the platform architecture in which the generic Adaptive KV core is separated from five runtime-facing roles: capability provider, runtime-family layer, runtime substrate (with retained exact Family A/B/C substrate implementations), replay backend, and composed runtime adapter. **Benchmark evidence in this appendix and the main report is anchored on the Llama Family A path**; Family B/C are retained exact production paths in code with their own narrower regression artifacts elsewhere in the repository.

## A.2 Software and Runtime Context

The primary retained evaluation artifacts use:

| Field | Value |
|---|---|
| Python | 3.13.5 |
| MLX | 0.31.1 |
| Platform | macOS / `darwin` |
| Model | `mlx-community/Llama-3.2-3B-Instruct-4bit` |

The benchmark harness drives the real `generate()` entry point rather than a synthetic inner-loop shim.

## A.3 Synthetic Benchmark Harness

The retained benchmark harness defines deterministic single-request scenarios. The main synthetic scenarios used in the final evaluation are:

| Scenario | Purpose |
|---|---|
| `long_static` | Long homogeneous context; mostly stable usage distribution |
| `topic_drift` | Topic A followed by topic B; stresses aging of earlier context |
| `delayed_topic_return` | A, then B, then A again; stresses recovery/promotion behavior |
| `hard_pressure_context` | Sized to force pressure under tighter budgets |

The canonical harness settings for the repeated final rerun were:

| Setting | Value |
|---|---|
| `prompt_target_tokens` | 512 |
| `decode_tokens` | 64 |
| `prefill_step_size` | 2048 |
| `clear_cache_interval` | 256 |
| `block_size_tokens` | 64 |
| `update_window_steps` | 16 |
| `seed` | 42 |

Two synthetic budget regimes were used:

| Regime | Soft budget | Hard budget | Baselines |
|---|---:|---:|---|
| SOFT / compression rerun | 42,000,000 | 250,000,000 | `non_adaptive`, `adaptive_full`, `adaptive_soft` |
| HARD / recovery rerun | 30,000,000 | 45,000,000 | `non_adaptive`, `adaptive_full`, `adaptive_hard` |

The retained final rerun used three repeats for adaptive baselines and median throughput aggregation per scenario.

## A.4 Real Workload Suite

The retained real workload suite contains ten repository-derived prompts:

| Group | Workloads | Characterization |
|---|---|---|
| Coding | `C1`–`C6` | Code- and implementation-oriented prompts derived from the MLXs repository |
| Technical writing / task prompts | `T1`–`T4` | Longer prose or instruction-heavy prompts with different structure and locality patterns |

Retained configuration for the main real-workload suite:

| Setting | Value |
|---|---|
| Repeats | 3 |
| Baselines | `non_adaptive`, `adaptive_full`, `adaptive_soft` |
| Decode tokens | 96 |
| Soft budget | 42,000,000 |
| Hard budget | 250,000,000 |

An additional HARD subset was also retained:

| Workload | Budget profile | Baseline |
|---|---|---|
| `C2` | hard stress | `adaptive_hard` |
| `T1` | hard stress | `adaptive_hard` |

## A.5 Correctness and Parity Checks

The evaluation uses several complementary correctness views:

| Signal | Meaning |
|---|---|
| `reference_token_match` | Greedy token identity to the non-adaptive reference for the same scenario |
| Focused parity reproductions | Small, repeatable prompt slices used to isolate specific parity issues |
| Unit tests | Scope invariants, replay behavior, compatibility gating, and resident-state semantics |

Important interpretation rule:

- `adaptive_full` is expected to match the non-adaptive greedy reference under the supported baseline.
- `adaptive_soft` preserves policy semantics and attention semantics over the resident state, but does not guarantee greedy token parity when compressed KV participates because quantization fidelity can perturb logits.
- `adaptive_hard` is judged primarily by exact replay-backed correctness and stable pressure behavior, not by parity with an all-resident throughput target.

## A.6 Metric Conventions

The most important reported quantities are:

| Metric | Meaning |
|---|---|
| `tokens_per_second` | End-to-end generation throughput for the single request |
| `adaptive_kv_recomputations_total` | Number of recompute requests, not replayed token count |
| `adaptive_kv_evictions_total` | Number of evictions observed in the run |
| `final_pressure_state` | Final pressure regime (`normal`, `soft`, `hard`) |
| `final_resident_bytes` | Resident adaptive KV bytes at the end of the run |
| `reference_token_match` | Greedy token equality versus reference |

## A.7 Interpreting Historical vs Final Artifacts

The repository contains multiple benchmark generations. They should not all be read as equivalent.

The retained interpretation used in the main report is:

1. Early one-shot synthetic artifacts are useful for understanding the system’s evolution and earlier bottlenecks.
2. The repeated final rerun is the principal evidence for the final V1 synthetic position.
3. The real-workload suite is the principal evidence for within-scope behavior on repository-derived prompts.
4. Later architecture-regression artifacts are read as regression gates, not as the primary performance claim for the system.

This separation matters because some older artifacts capture intermediate implementation stages, and some later one-shot regression checks are intentionally narrow and noise-sensitive.
