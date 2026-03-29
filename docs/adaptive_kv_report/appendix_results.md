# Appendix B. Results Tables

> **Archival note:** This appendix belongs to the Adaptive KV **V1** / `FULL`-`COMPRESSED`-`EVICTED` evaluation-era report. It is historical source material. For the **frozen** exact adaptive execution-pack baseline and archived track status, see [`../adaptive_kv_turboquant_first.spec.md`](../adaptive_kv_turboquant_first.spec.md) and [`../adaptive_kv_execution_pack_track_archival.md`](../adaptive_kv_execution_pack_track_archival.md).

## B.1 Final Repeated Synthetic Rerun: SOFT Regime

Median throughput over three repeats for the final SOFT rerun:

| Scenario | `adaptive_full` tok/s | `adaptive_soft` tok/s | Soft / Full |
|---|---:|---:|---:|
| `long_static` | 211.55 | 207.89 | 0.98 |
| `topic_drift` | 203.81 | 198.66 | 0.97 |
| `delayed_topic_return` | 197.63 | 201.58 | 1.02 |
| `hard_pressure_context` | 215.51 | 210.13 | 0.98 |

All adaptive rows in this rerun reported `reference_token_match = true`. SOFT runs stayed at zero evictions and zero recompute requests.

## B.2 Final Repeated Synthetic Rerun: HARD Regime

Median throughput over three repeats for the final HARD rerun:

| Scenario | `adaptive_full` tok/s | `adaptive_hard` tok/s | Hard / Full | Evictions | Recompute requests | Final pressure | Final resident bytes |
|---|---:|---:|---:|---:|---:|---|---:|
| `long_static` | 216.28 | 148.51 | 0.69 | 1 | 1 | `soft` | 41,058,304 |
| `topic_drift` | 216.14 | 153.07 | 0.71 | 1 | 1 | `soft` | 41,058,304 |
| `delayed_topic_return` | 211.01 | 151.67 | 0.72 | 1 | 1 | `soft` | 41,058,304 |
| `hard_pressure_context` | 211.97 | 149.32 | 0.70 | 1 | 1 | `soft` | 41,058,304 |

All adaptive rows in this rerun reported `reference_token_match = true`.

## B.3 Mixed-Tier Usage-Path Bottleneck and Deferred Extraction

Representative profiler-style comparison for the late usage-path optimization:

| Case | Throughput | Usage-path share of wall |
|---|---:|---:|
| Earlier mixed-tier representative run | 90.3 tok/s | 72.5% |
| After deferred usage extraction | 196.9 tok/s | 1.13% |
| All-FULL reference in same rerun | 211.0 tok/s | n/a |

The optimization preserved the observer signal and changed only when materialization occurred.

## B.4 Before/After View for the Late V1 Optimization Sequence

These values summarize the improvement from earlier one-shot diagnostics to the repeated final rerun:

| Scenario | Earlier `adaptive_soft` tok/s | Final `adaptive_soft` tok/s | Earlier `adaptive_hard` tok/s | Final `adaptive_hard` tok/s |
|---|---:|---:|---:|---:|
| `long_static` | 65.93 | 207.89 | 66.12 | 148.51 |
| `topic_drift` | 76.65 | 198.66 | 68.26 | 153.07 |
| `delayed_topic_return` | 78.72 | 201.58 | 65.19 | 151.67 |
| `hard_pressure_context` | 77.40 | 210.13 | 67.75 | 149.32 |

These gains came from a sequence of exact runtime improvements, especially compressed-run coalescing, deferred usage extraction, replay-path hardening, and later targeted cleanup of remaining overheads.

## B.5 Real Workload Suite Medians

Median throughput across three repeats per workload:

| Workload | `non_adaptive` | `adaptive_full` | `adaptive_soft` | Soft / Full |
|---|---:|---:|---:|---:|
| `C1` | 211.93 | 208.82 | 197.64 | 0.95 |
| `C2` | 164.07 | 162.15 | 154.11 | 0.95 |
| `C3` | 209.55 | 208.34 | 203.74 | 0.98 |
| `C4` | 211.75 | 217.31 | 200.82 | 0.92 |
| `C5` | 79.38 | 78.71 | 78.14 | 0.99 |
| `C6` | 202.10 | 202.38 | 203.87 | 1.01 |
| `T1` | 212.32 | 213.02 | 213.82 | 1.00 |
| `T2` | 149.84 | 141.85 | 151.57 | 1.07 |
| `T3` | 286.92 | 276.56 | 268.70 | 0.97 |
| `T4` | 295.99 | 296.61 | 291.74 | 0.98 |

Across these ten workloads, `adaptive_soft / adaptive_full` ranged from approximately `0.92` to `1.07`.

In this retained real-workload snapshot:

- `adaptive_full` showed no greedy-token mismatches against the non-adaptive reference.
- `adaptive_soft` mismatches were present only on `C1` and `C2`.

## B.6 Real-Workload HARD Subset

Representative HARD-stress real workloads:

| Workload | Median tok/s | Evictions | Recompute requests | Final pressure | Final resident bytes | `reference_token_match` |
|---|---:|---:|---:|---|---:|---|
| `C2` | 124.48 | 1 | 1 | `soft` | 39,221,504 | true |
| `T1` | 66.59 | 22 | 5 | `hard` | 60,628,736 | true |

The HARD subset is intentionally stress-oriented and should be read as evidence of semantic robustness under pressure, not as the normal target operating regime.

## B.7 Focused Parity Reproductions

Focused checked-in reproductions currently report no first-difference on the retained prompt slices:

| Artifact | Seeds | `adaptive_full` first diff vs reference | `adaptive_soft` first diff vs reference |
|---|---:|---|---|
| `C5` | 42, 43, 44 | none | none |
| `T4` | 42, 43, 44 | none | none |
| `C5` prefix slice | 42 | none | none |
| `T4` prefix slice | 42 | none | none |

These reproductions are useful as tight regression checks, but they do not supersede the broader scope statement: `adaptive_soft` still does not promise greedy parity while compressed KV is active.

## B.8 Runtime-Families Architecture Regression Checks

Later same-machine spot checks used during the runtime-families architecture program kept the retained Llama path as the Family A control gate.

The current-tree spot checks were:

| Case | Adaptive tok/s | Evictions | Recompute requests | Final pressure | Final resident bytes | `reference_token_match` |
|---|---:|---:|---:|---|---:|---|
| SOFT `long_static` `adaptive_soft` | 187.98 | 0 | 0 | `soft` | 44,613,632 | true |
| HARD 512 `hard_pressure_context` `adaptive_hard` | 182.62 | 1 | 1 | `soft` | 41,058,304 | true |
| HARD 1024 `hard_pressure_context` `adaptive_hard` | 196.77 | 6 | 1 | `hard` | 71,335,936 | true |

The more stable architectural readout is still the adaptive/full ratio against earlier retained sanity artifacts:

| Case | Earlier sanity adaptive / full | Runtime-family run adaptive / full | Interpretation |
|---|---:|---:|---|
| SOFT `adaptive_soft / adaptive_full` | 0.939 | 0.902 on first run, 1.137 on immediate rerun | One-shot SOFT signal was noisy and not treated as evidence of regression |
| HARD 512 `adaptive_hard / adaptive_full` | 0.919 | 0.949 | Stable to slightly improved |
| HARD 1024 `adaptive_hard / adaptive_full` | 0.726 | 0.721 | Effectively unchanged |

These checks were used as regression gates, not as the primary synthetic performance claim for the system. Their purpose was to decide whether the explicit runtime-family layer could be retained without a credible material slowdown on the Family A control path.

## B.9 Runtime-Family Support Outcomes

The retained runtime-family architecture makes the support boundary explicit:

| Model/runtime shape | Runtime family | Retained status |
|---|---|---|
| Llama retained path | Family A `full_kv` | Supported control/baseline path (principal benchmark surface in this report) |
| Full-attention-only `qwen3_5` subset | Family A `full_kv` | Supported when homogeneous `list[KVCache]` baseline and other gates pass |
| Standard text-only `qwen3_5` | Family C `hybrid_state` | Supported when hybrid layer/cache contract passes the gate |
| Full-attention-only `ministral3` subset | Family A `full_kv` | Supported when homogeneous `list[KVCache]` baseline and other gates pass |
| Standard `ministral3` (sliding-window layers) | Family B `windowed_kv` | Supported when `sliding_window`, layer alignment, and `KVCache`/`RotatingKVCache` contract pass the gate |
| Multimodal `qwen3_5` or `ministral3` | — | Unsupported (`input_embeddings` paths) |
| `compile_decode`, legacy `quantized_kv_start`, external cache reuse | — | Unsupported |

The checked-in `family_subset_micro.json` artifact remains useful as tight smoke evidence for Family A-compatible subsets; separate artifacts (for example under `results/bench_adaptive/`) document Family B/C regression-style runs and must not be over-interpreted as universal production throughput claims.
