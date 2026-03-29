# Adaptive KV for MLXs: Exact Adaptive KV-Cache Management for Single-Request Text Decoder Inference

> **Archival note**
>
> This report is historical source material for Adaptive KV **V1** / `FULL`-`COMPRESSED`-`EVICTED` evaluation-era benchmarks and conclusions. It does not describe the later **exact adaptive execution-pack** frozen baseline, and it does not define the current **mainline pivot** toward pragmatic `FULL` / `COMPRESSED` / `EVICTED`.
>
> For the execution-pack baseline (technical freeze, track archival, throughput limits, next direction), use [../adaptive_kv_execution_pack_track_archival.md](../adaptive_kv_execution_pack_track_archival.md), [../adaptive_kv_turboquant_first.spec.md](../adaptive_kv_turboquant_first.spec.md), and [../dev/repo_mapping_kv_cache.md](../dev/repo_mapping_kv_cache.md).

## Abstract

Transformer decoding with append-only KV retention has the simplest semantics but the worst possible memory growth profile: every past token is treated as equally valuable until the context window is exhausted. Fixed sliding windows bound memory, but they do so by imposing a rigid recency prior that can discard semantically important history. Adaptive KV in MLXs addresses this problem by treating KV residency as a memory-management problem rather than an attention-rewrite problem.

The retained system operates at block granularity, moves blocks through three tiers (`FULL`, `COMPRESSED`, `EVICTED`), preserves resident ordering exactly, and restores evicted history through explicit replay from authoritative source-token spans. It is observer-only: it never edits queries, keys, values, logits, or sampling policy. The product boundary stays intentionally narrow: single-request generation only, honest capability gating, and three retained **exact** concrete substrates—Family A `full_kv`, Family B `windowed_kv`, and Family C `hybrid_state`. The semantic core remains generic; model adapters classify the runtime and bind into the matching family substrate. The verified Llama path remains the primary Family A control and regression gate. Standard text-only `qwen3_5` is retained on Family C when the hybrid cache layout matches the adapter contract; full-attention-only `qwen3_5` subsets remain on Family A where applicable. Standard `ministral3` with sliding-window layers is retained on Family B when layer types, `sliding_window`, and per-layer `KVCache` / `RotatingKVCache` alignment satisfy the gate. Multimodal `qwen3_5`, multimodal `ministral3` (`input_embeddings` paths), `compile_decode`, legacy `quantized_kv_start`, and external cache reuse remain unsupported.

Within that scope, the final evidence is strong for the **principal Llama Family A benchmark suite** documented below. In repeated synthetic SOFT-regime runs, `adaptive_soft` reaches `0.97x` to `1.02x` the throughput of `adaptive_full` across the retained scenario set while keeping zero evictions and zero replay requests. In repeated HARD-regime runs, `adaptive_hard` is materially slower than `adaptive_full`, but it remains correct, replay-backed, and stable under pressure, with a remaining throughput gap that is structural to the exact mixed-tier segmented executor rather than a sign of incorrect semantics. Real-workload medians across ten repository-derived prompts place `adaptive_soft` between `0.92x` and `1.07x` of `adaptive_full`. The retained implementation baseline is a runtime-family architecture in which the generic Adaptive KV core is separated from an explicit capability provider, runtime-family layer, **three** retained exact runtime substrates (A/B/C), replay backend, and composed runtime adapter. Family B and Family C are no longer architectural placeholders: they are exercised production paths where the capability gate reports full support.

## 1. Introduction

Long-context autoregressive inference turns the KV cache into a first-class systems problem. The cost is not merely that keys and values consume memory; it is that append-only retention couples semantic history to the worst possible memory-growth policy. All past context is retained at equal fidelity until memory pressure or context limits force a crude intervention.

Adaptive KV addresses that problem without changing the transformer’s semantics. Instead of altering attention itself, the system changes the residency of previously computed KV state. Some blocks remain full-fidelity, some are retained in a compressed resident form, and some are evicted altogether. When evicted context becomes necessary again, it is reconstructed by replaying the authoritative source-token span through the model rather than by inventing an approximate surrogate.

That design choice makes the systems problem harder, not easier. It introduces a control plane, a resident-state representation, recovery logic, and mixed-tier attention execution. The work reported here is therefore not just a policy sketch. It is a retained implementation in MLXs, with repeated synthetic benchmarks, focused parity investigations, real-workload evaluation, and a later sequence of architecture programs that produced a **multi-family** runtime baseline (Families A/B/C) with further product surface still explicitly out of scope.

This document presents the final retained system as a standalone technical report. It separates the design itself from the implementation choices, the guarantees from the non-guarantees, and the measured results from the later engineering programs that hardened and reorganized the implementation.

## 2. Problem Statement and Design Goals

The underlying question is straightforward:

Can a KV-cache residency policy improve memory behavior over append-only retention and fixed sliding windows while preserving exact transformer semantics over the resident state and preserving correct behavior under replay-backed recovery?

The retained design goals were:

1. Preserve attention semantics rather than rewrite them.
2. Make retention adaptive at block granularity rather than by fixed recency windows.
3. Use real replay-backed recovery for evicted content.
4. Keep the policy observer-only.
5. Maintain explicit live-byte budgeting and pressure handling.
6. Stay honest about support scope instead of claiming broad model compatibility.

The retained support envelope is intentionally narrow:

| Dimension | Retained position |
|---|---|
| Generation mode | Single-request generation only |
| Runtime-family substrates | Retained exact substrates for Family A `full_kv`, Family B `windowed_kv`, and Family C `hybrid_state` |
| Semantic core vs substrates | Generic core; adapters classify the runtime and bind into the family-specific substrate |
| Primary control / benchmark path | Verified Llama path within Family A `full_kv` |
| Family B retained production path | Standard `ministral3` when sliding-window layers and the per-layer `KVCache` / `RotatingKVCache` contract pass the gate |
| Family C retained production path | Standard text-only `qwen3_5` when hybrid linear-attention layers and the per-layer `KVCache` / `ArraysCache` contract pass the gate |
| Family A beyond Llama | Full-attention-only `qwen3_5` and `ministral3` subsets that satisfy the homogeneous `list[KVCache]` baseline |
| Baseline cache shape | Family A: homogeneous `list[KVCache]`; Families B/C: gated heterogeneous layouts per adapter rules |
| Tiers | `FULL`, `COMPRESSED`, `EVICTED` |
| Recovery | Real replay-backed recovery |
| Observer semantics | Required |
| Unsupported in V1 | Multimodal `qwen3_5`, multimodal `ministral3`, `compile_decode`, legacy `quantized_kv_start`, external cache reuse, configurations rejected by the capability gate, and other unsupported model families |

This report therefore describes a bounded exact system, not a generic cache abstraction that is claimed to work for every model family.

## 3. System Design

### 3.1 Block-Level Residency Model

The prompt and generated history are partitioned into logical blocks of fixed token width. Each block carries:

- a logical token span,
- a source-token span used for replay,
- a tier assignment,
- bounded score components used by the policy,
- and transition metadata used to enforce dwell and cooldown rules.

Each block is in exactly one of three resident tiers:

- `FULL`: resident full-fidelity KV state,
- `COMPRESSED`: resident quantized KV state,
- `EVICTED`: no resident KV state, only metadata and replay provenance.

The ordinary transition direction is:

`FULL -> COMPRESSED -> EVICTED`

Promotion in the opposite direction is allowed when justified, but an evicted block cannot become resident again without replay-backed reconstruction.

### 3.2 Control Plane

Adaptive KV updates policy state at fixed decode windows rather than on every token. At each policy window the retained implementation:

1. collects usage accumulated over the window,
2. updates the block scores,
3. computes the current pressure regime from live resident bytes,
4. applies score-guided promotions and demotions subject to protection rules,
5. then evicts additional candidates if the current budget still requires it.

This is not a pure score sorter. The score proposes an ordering of utility; the budget and protection rules determine which transitions are currently legal.

### 3.3 Resident State and Segments

Attention does not consume a logical block table directly. It consumes an ordered resident state assembled from the non-evicted blocks in logical order. The retained implementation exposes that resident state as a sequence of attention segments:

- contiguous `FULL` runs,
- coalesced compressed runs,
- and resident-order slice metadata used both by the attention path and by usage attribution.

The resident state is therefore a data-plane view of the current policy state: it is derived from the logical block registry, but arranged for attention execution.

### 3.4 Recovery Barrier

Before a decode forward that requires historical context, the manager checks whether any logically required blocks are currently evicted. If so, recovery is triggered before the forward is allowed to proceed.

The retained system uses replay as the recovery mechanism. It does not invent approximate resident KV for evicted blocks. Replay uses the authoritative source-token spans and the same model path, then reinserts the reconstructed resident state.

### 3.5 Operating Regimes

The retained system is best understood as three operating regimes:

- **FULL / normal pressure:** all required state stays resident at full fidelity.
- **SOFT:** compression is active, but replay pressure is absent or rare. This is the target operating regime for within-scope performance.
- **HARD:** budgets are tight enough that replay-backed recovery occurs during decode. This regime remains correct and useful, but it is not expected to match an all-resident baseline in throughput.

## 4. Formal and Mathematical Model

### 4.1 Blocks, Tiers, and Resident State

Let the sequence history be partitioned into logical blocks \( b_1, \dots, b_n \), each with a token span and an authoritative source span. Let the tier of block \( b \) at decode step \( t \) be


\[
\tau_b^{(t)} \in \{\text{FULL}, \text{COMPRESSED}, \text{EVICTED}\}.
\]

Each non-evicted block contributes a resident representation \( r_b^{(t)} \):

\[
r_b^{(t)} =
\begin{cases}
(K_b^{(t)}, V_b^{(t)}) & \tau_b^{(t)} = \text{FULL} \\
(\tilde K_b^{(t)}, \tilde V_b^{(t)}) & \tau_b^{(t)} = \text{COMPRESSED} \\
\varnothing & \tau_b^{(t)} = \text{EVICTED}
\end{cases}
\]

The resident state at time \( t \) is the ordered concatenation of all non-evicted block representations:

\[
\mathcal{R}^{(t)} = \operatorname{concat}(r_b^{(t)} \mid \tau_b^{(t)} \neq \text{EVICTED}).
\]

The crucial invariant is order preservation: the resident state must expose the same logical ordering as the original history for all non-evicted content.

### 4.2 Retention Score

The retained policy is score-guided, not score-only. Each block maintains a bounded state summarized by the composite score

\[
S_b = w_h H_b + w_p P_b + w_s R_b - w_a A_b,
\]

where:

- \( H_b \) is recent hotness,
- \( P_b \) is longer-timescale persistence,
- \( R_b \) is a bounded structural prior,
- \( A_b \) is age or staleness,
- and \( w_h, w_p, w_s, w_a \) are bounded policy weights.

Usage enters the score through a windowed observer signal \( U_b^{(t)} \). Conceptually, the retained update family is:

\[
H_b^{(t+1)} = \rho_h H_b^{(t)} + (1-\rho_h) U_b^{(t)},
\]

\[
P_b^{(t+1)} = \rho_p P_b^{(t)} + (1-\rho_p) U_b^{(t)},
\]

\[
A_b^{(t+1)} = A_b^{(t)} + \lambda_a \cdot \mathbf{1}[U_b^{(t)} = 0].
\]

This form matters because it gives the policy graduality and recoverability. Blocks cool down rather than disappearing instantly; structural priors can protect important spans; and previously cold blocks can recover if their observed usage rises again.

### 4.3 Budget-Aware Transition Logic

Let \( M^{(t)} \) be the live resident bytes at time \( t \), with budgets \( B_{\text{soft}} \) and \( B_{\text{hard}} \).

The policy regime is:

\[
P^{(t)} =
\begin{cases}
\text{normal} & M^{(t)} < B_{\text{soft}} \\
\text{soft} & B_{\text{soft}} \le M^{(t)} < B_{\text{hard}} \\
\text{hard} & M^{(t)} \ge B_{\text{hard}}.
\end{cases}
\]

The transition logic is therefore not “evict the lowest score” in the abstract. It is:

- maintain or promote valuable blocks when budget allows,
- demote more aggressively in `soft`,
- and, in `hard`, continue moving memory downward unless only protected or otherwise unavoidable blocks remain.

### 4.4 Exact Mixed-Tier Attention

Suppose the resident state is presented as ordered segments \( \{(K_j, V_j)\}_{j=1}^{m} \). For a query tensor \( Q \), the retained mixed-tier attention path computes segment-local scores

\[
Z_j = Q K_j^\top
\]

and concatenates them along the key axis:

\[
Z = \operatorname{concat}(Z_1, \dots, Z_m).
\]

A single global mask \( M \) and a single global softmax are applied:

\[
\alpha = \operatorname{softmax}(Z + M).
\]

If segment \( j \) occupies resident columns \( [s_j, e_j) \), then the output is

\[
O = \sum_{j=1}^{m} \alpha_{[..., s_j:e_j]} V_j.
\]

This is the key semantic claim: the retained executor may be segmented internally, but it computes attention over the assembled resident state exactly. The system cost of segmentation can be high, especially in HARD steady state, but the formulation does not approximate the attention result over the current resident state.

### 4.5 Replay Condition

If decode requires a block \( b \) such that \( \tau_b^{(t)} = \text{EVICTED} \), then the system performs:

\[
\operatorname{recover}(b) := \operatorname{replay}(\text{source span of } b)
\]

before the forward that consumes that history. The replay uses authoritative source tokens and the retained model path; the reconstructed KV is then reinserted as resident state. The result is exact recovery with respect to the retained execution path, not an approximation or statistical guess.

## 5. Implementation in MLXs

### 5.1 Generic Core

The retained MLXs implementation now separates model-agnostic Adaptive KV semantics from model- and runtime-specific bindings.

The generic core owns:

- block metadata and registry management,
- usage aggregation,
- score computation,
- tier transitions,
- ghost and recompute coordination,
- pressure-state tracking,
- metrics and diagnostics,
- and the top-level orchestration of policy windows and recovery barriers.

This core is intended to remain stable even if additional model families are introduced later.

### 5.2 Runtime-Family Platform Structure

The retained runtime boundary is now explicit and organized around runtime families. The generic semantic core remains stable, while runtime-specific behavior is factored into a family layer and model adapters. The retained architecture separates five roles:

1. **Capability provider**  
   Declares whether a model/runtime configuration is fully supported, partially supported, or unsupported, and reports the reason honestly.
2. **Runtime-family layer**  
   Classifies the concrete runtime shape into an explicit family and supplies the corresponding family bindings. The currently explicit families are Family A `full_kv`, Family B `windowed_kv`, and Family C `hybrid_state`.
3. **Runtime substrate**  
   Owns per-layer resident state, live-byte accounting, resident-state construction for attention, and tier-local storage behavior.
4. **Replay backend**  
   Owns scratch replay state, prefix extension, and replay-token extraction.
5. **Composed runtime adapter**  
   Binds the capability provider and resolved family bindings into a concrete model/runtime implementation.

The manager now orchestrates abstract runtime and replay components together with resolved family bindings rather than depending on a Llama-specific architectural assumption.

### 5.3 Runtime Families and Retained Production Bindings

The runtime-family layer distinguishes three concrete substrates; each is a **retained exact** implementation path where the capability gate accepts the configuration.

- **Family A `full_kv`** — homogeneous token-addressable `KVCache` runtimes across layers. The verified Llama path remains the primary control/baseline and the main documented benchmark surface. Full-attention-only compatible subsets of `qwen3_5` and `ministral3` can bind here when they present a homogeneous `list[KVCache]` baseline.
- **Family B `windowed_kv`** — KV-based runtimes whose resident semantics depend on local, sliding, or rotating windows. Standard `ministral3` with sliding layers binds here: the substrate preserves window semantics while Adaptive KV manages tiering, resident assembly, and replay on the supported path.
- **Family C `hybrid_state`** — runtimes that mix linear-attention state with full-attention KV. Standard **text-only** `qwen3_5` binds here when layers and cache entries align (`KVCache` on full-attention layers, `ArraysCache` on linear layers). Multimodal requests (`input_embeddings` present) remain unsupported for this adapter.

The capability model still prevents silent partial support: misaligned cache shapes, missing `sliding_window` where required, `compile_decode`, legacy `quantized_kv_start`, external cache reuse, and unsupported model types are rejected with explicit reasons.

### 5.4 Resident Storage and Attention Paths

The resident data plane uses:

- full-fidelity per-layer resident storage for `FULL` blocks,
- exact-sized quantized compressed-run stores for `COMPRESSED` runs,
- and ordered resident-segment metadata for attention and usage attribution.

The retained attention implementation preserves the fast all-FULL path, preserves the single-compressed fast path where applicable, and keeps the generic segmented path as the exact oracle and fallback. Earlier runtime work also retained executor-ready resident slice metadata and a score-side decode specialization for `q_len = 1` mixed-tier decode, but rejected broader executor-fusion ideas that were benchmark-negative.

## 6. Correctness and Semantic Safety

### 6.1 Observer-Only Semantics

Adaptive KV in the retained system is observer-only. It does not alter:

- query tensors,
- key tensors,
- value tensors,
- logits,
- or decoding policy.

Policy decisions are driven by observed usage and live-byte budgets, not by modifying the transformer computation.

### 6.2 Recovery Safety

The final system enforces a genuine required-resident barrier before decode. An evicted block cannot silently re-enter resident state, and decode does not proceed through missing history in the hope that later repair will make the result consistent. If required history is absent, replay happens first.

### 6.3 Exactness vs Fidelity

The retained correctness envelope is intentionally precise:

| Mode | Retained statement |
|---|---|
| `adaptive_full` | Expected to match the non-adaptive greedy reference under each **fully supported** gated configuration; the verified Llama Family A path remains the principal documented benchmark case |
| `adaptive_soft` | Preserves adaptive policy semantics and exact mixed-tier attention over the resident state, but does not promise greedy token parity when compressed KV participates |
| `adaptive_hard` | Preserves replay-backed semantic correctness under pressure, but is not expected to match all-resident throughput |

This distinction matters. A soft-only token difference under active compression is not the same failure mode as a replay bug or a resident-order bug.

### 6.4 Support Honesty

The retained runtime-family architecture exposes support explicitly through a capability model with `FULL`, `PARTIAL`, and `UNSUPPORTED` outcomes together with explicit runtime-family classification. Unsupported families and unsupported configurations fail clearly instead of appearing to work accidentally.

## 7. Evaluation Methodology

The evaluation combines four complementary evidence streams:

1. repeated synthetic scenario benchmarks,
2. focused profiler and diagnostic studies,
3. real workload evaluation on repository-derived prompts,
4. and later regression checks used to accept or reject architectural refactors.

The main synthetic scenarios are `long_static`, `topic_drift`, `delayed_topic_return`, and `hard_pressure_context`. The final repeated rerun uses 512-token prompts and 64-token decode targets. The SOFT rerun uses a `42 MB / 250 MB` soft/hard budget pair; the HARD rerun uses `30 MB / 45 MB`.

The real-workload suite uses ten repository-derived prompts (`C1`–`C6`, `T1`–`T4`) with three repeats per workload and 96 generated tokens per run. A smaller HARD subset was also retained for stress-oriented real prompts.

Detailed setup, conventions, and scope are given in [Appendix A](./appendix_experimental_setup.md).

## 8. Experimental Results

### 8.1 Bottleneck Isolation and Deferred Usage Extraction

One of the most important late-stage findings was that the dominant mixed-tier overhead in the earlier implementation was not the score engine itself, but the observer path used to materialize per-block usage. On a representative coalesced mixed-tier case, usage extraction accounted for roughly `72.5%` of wall time and throughput sat near `90.3 tok/s`.

The retained fix did not change the observer signal. It changed when that signal was materialized. Usage values are now kept on device and flushed at the policy-window boundary rather than forcing synchronization and host-side scalar extraction on every mixed-tier forward. In the corresponding profiler rerun, the usage-flush portion dropped to roughly `1.13%` of wall time and throughput rose to about `196.9 tok/s`, while an all-FULL reference in the same rerun was about `211.0 tok/s`.

This was the decisive systems optimization that moved the SOFT regime from structurally bottlenecked to practically competitive.

### 8.2 SOFT Regime

The final repeated synthetic rerun shows that the main target regime, compression without replay-heavy stress, is now close to the all-resident baseline:

| Scenario | `adaptive_full` tok/s | `adaptive_soft` tok/s | Soft / Full |
|---|---:|---:|---:|
| `long_static` | 211.55 | 207.89 | 0.98 |
| `topic_drift` | 203.81 | 198.66 | 0.97 |
| `delayed_topic_return` | 197.63 | 201.58 | 1.02 |
| `hard_pressure_context` | 215.51 | 210.13 | 0.98 |

SOFT runs in this rerun remained at zero evictions and zero recompute requests, and all adaptive rows reported `reference_token_match = true`. The system therefore reached its most important within-scope target: adaptive compression is no longer paying a large avoidable systems tax in the normal operating regime.

### 8.3 HARD Regime

The final HARD rerun tells a different but still useful story:

| Scenario | `adaptive_full` tok/s | `adaptive_hard` tok/s | Hard / Full |
|---|---:|---:|---:|
| `long_static` | 216.28 | 148.51 | 0.69 |
| `topic_drift` | 216.14 | 153.07 | 0.71 |
| `delayed_topic_return` | 211.01 | 151.67 | 0.72 |
| `hard_pressure_context` | 211.97 | 149.32 | 0.70 |

The discrete recovery pattern is stable across these scenarios: one eviction, one recompute request, final pressure state `soft`, final resident bytes `41,058,304`, and `reference_token_match = true`.

The correct interpretation is not that HARD is “broken.” It is that the remaining throughput gap is structural to the exact replay-backed and mixed-tier segmented execution path under small budgets. The final system treats this as acceptable within scope, not as an unfinished ordinary hardening item.

### 8.4 Real Workloads

Real-workload medians confirm the same qualitative picture. Across ten repository-derived prompts, `adaptive_soft / adaptive_full` ranges from approximately `0.92` to `1.07`, with many workloads nearly indistinguishable in throughput:

| Workload | `adaptive_full` tok/s | `adaptive_soft` tok/s | Soft / Full |
|---|---:|---:|---:|
| `C1` | 208.82 | 197.64 | 0.95 |
| `C5` | 78.71 | 78.14 | 0.99 |
| `T1` | 213.02 | 213.82 | 1.00 |
| `T4` | 296.61 | 291.74 | 0.98 |

The detailed table is provided in [Appendix B](./appendix_results.md). In that retained snapshot, `adaptive_full` showed no greedy-token mismatches against the non-adaptive reference. `adaptive_soft` mismatches were limited to `C1` and `C2`.

Historical focused investigations separated two correctness stories:

- an `adaptive_full` parity bug on `C5`, which was fixed by restoring the fused all-FULL forward path while still deriving usage information safely,
- and a soft-only compression-sensitive drift classification on `T4`, which matters because it reflects quantized KV fidelity rather than mixed-tier control failure.

The currently checked-in focused reproductions for `C5` and `T4` do not show a first-difference on the retained prompt slices, but the scope guarantee remains unchanged: `adaptive_soft` does not promise greedy parity while compressed KV is active.

### 8.5 Runtime-Family Architecture Regression Checks

Later architecture work retained the runtime-family layer only if the Family A control path stayed within benchmark noise. Same-machine spot checks on the retained Llama path did not show a credible material regression. The first SOFT one-shot moved from `0.939` to `0.902` on the adaptive/full ratio, but an immediate rerun reversed direction to `1.137`, so the SOFT absolute signal was treated as noisy. The harder cases were more stable:

| Case | Earlier sanity ratio | Runtime-family run ratio |
|---|---:|---:|
| SOFT `adaptive_soft / adaptive_full` | 0.939 | 0.902 on first run, 1.137 on rerun |
| HARD 512 `adaptive_hard / adaptive_full` | 0.919 | 0.949 |
| HARD 1024 `adaptive_hard / adaptive_full` | 0.726 | 0.721 |

That is the relevant architectural reading for the **Family A Llama control path**: no semantic regression signal, no stable material slowdown versus earlier sanity ratios, and throughput on SOFT one-shots treated as machine-noise-sensitive. Subsequent work completed retained **exact** Family B and Family C substrates and wired standard `ministral3` / text-only `qwen3_5` into those paths under the same capability discipline; the synthetic tables in this report remain Llama-centric and should not be read as performance claims for B/C unless backed by separate artifacts.

## 9. Engineering Evolution and Validation Trajectory

The final system became credible through a small number of turning points rather than through one monolithic design step.

| Milestone | Key issue | Change | Why it mattered |
|---|---|---|---|
| Initial integrated version | Adaptive logic existed, but resident visibility and stress behavior were not yet fully trustworthy | Integrated block registry, transitions, scoring, and mixed-tier execution into MLXs | Established the end-to-end system shape |
| Replay-backed hardening | Evicted content could not be treated as logically present without real reconstruction | Added required-resident barriers and replay-backed recovery | Turned HARD behavior into a semantically serious mode rather than a heuristic one |
| Compressed-run coalescing | Fragmented resident state imposed avoidable mixed-tier overhead | Coalesced adjacent compressed blocks into runs | Reduced resident fragmentation cost without changing semantics |
| Deferred usage extraction | Observer materialization dominated mixed-tier wall time | Moved usage batching and materialization to policy-window flush | Shifted SOFT from clearly bottlenecked to near-`adaptive_full` performance |
| Parity restoration and classification | Different correctness issues were being conflated | Restored the fused all-FULL path for `adaptive_full` and separated compression-sensitive drift from control bugs | Clarified the real guarantee boundary |
| Runtime-families architecture program | The retained implementation was correct but still organized too much around model-local substrate ownership | Separated the generic core from an explicit runtime-family layer and moved model adapters to family classification and binding | Produced the retained multi-family baseline |
| Family B/C production substrates | Families B and C were classified but needed real exact substrates, not Family A assumptions | Implemented `windowed_kv` and `hybrid_state` runtime substrates and adapter bindings for standard `ministral3` and text-only `qwen3_5` | Adaptive KV V1 is fully implemented for A/B/C within the stated unsupported-feature boundary; the project holds a true multi-family retained baseline |

## 10. Discussion

Three conclusions dominate the final evidence.

First, the design is semantically viable. Exact replay-backed recovery, observer-only usage collection, budget-aware residency transitions, and exact mixed-tier attention over the resident state are compatible in a real implementation.

Second, the main target regime is now in the right performance band **on the principal Llama Family A evidence in this report**. The final repeated SOFT rerun and the retained real-workload suite both show `adaptive_soft` behaving close to `adaptive_full` in that scope. This is the key practical outcome of the documented benchmark surface.

Third, the remaining HARD gap is no longer best interpreted as a control bug, a replay-safety bug, or an obvious Python overhead bug. Earlier bottlenecks of those kinds were found and addressed. What remains is structural to the retained exact segmented executor and replay-backed stress path.

The later runtime and architecture programs reinforce that reading. Broader executor fusion, deeper exact-runtime redesigns, and runtime-architecture prototypes were explored seriously and benchmarked, but they either regressed or did not clear the retention threshold. The runtime-families program reorganized support around explicit families; on the Family A Llama control path it did not show a credible stable material regression, and **exact** Family B/C substrates were then completed so the architecture is no longer “Family A plus placeholders.” The system is at a principled stopping point for V1 within the explicit unsupported-feature list, not merely a convenient one.

## 11. Limitations and Non-Goals

The retained limitations are explicit:

- single-request generation only,
- multimodal `qwen3_5`, multimodal `ministral3`, `compile_decode`, legacy `quantized_kv_start`, and external cache reuse are out of scope,
- configurations that fail the capability gate (including cache shape or architecture mismatches) are unsupported even when the model family is nominally Llama, `ministral3`, or `qwen3_5`,
- no guarantee of greedy token parity for `adaptive_soft` while compressed KV participates,
- no goal of matching `adaptive_full` throughput under HARD pressure,
- principal synthetic and real-workload tables in this report remain **Llama Family A** evidence; they do not by themselves characterize Family B/C throughput,
- and no claim that the current exact mixed-tier segmented executor is optimal under hard stress.

These are not oversights in the presentation. They are part of the system boundary.

## 12. Future Work

Any future work should be driven by explicit product or platform needs rather than by the existence of remaining engineering curiosity.

The most plausible future directions are:

1. **Compressed-KV fidelity work**  
   If tighter soft-mode parity is required under active compression, the natural target is quantized KV fidelity rather than adaptive-policy control.
2. **Optional deeper HARD executor R&D**  
   The remaining HARD gap is structural to the current exact mixed-tier segmented executor and replay-backed stress path. Any further improvement there should be treated as dedicated R&D, not routine hardening.
3. **Broader product surface**  
   Natural extensions—multimodal generation paths, `compile_decode` integration, legacy `quantized_kv_start`, external cache reuse, or additional model families—require explicit design and gating beyond the current retained envelope.

## 13. Conclusion

Adaptive KV for MLXs is now a retained, bounded exact system rather than a speculative design. It manages KV residency at block granularity, preserves attention semantics over the resident state, uses real replay-backed recovery for evicted content, and exposes support honestly through an explicit runtime-family architecture with **three retained exact substrates** (Families A, B, and C).

Within its approved scope, the result is clear. `adaptive_full` is aligned with the supported non-adaptive baseline on fully gated configurations; `adaptive_soft` is performant enough in the normal compression regime to be practically useful on the **principal Llama benchmark suite** reported here; and `adaptive_hard` is semantically safe and good enough under stress even though its remaining throughput gap is structural. Architecture refactors strengthened the boundary without a credible stable regression on the Family A Llama control path, and correctness remained exact as Family B/C moved from classification-only to retained production paths.

That is the right final position for V1: not universal support, not maximal throughput under every pressure regime, but a serious exact **multi-family** implementation with clear guarantees, honest limits, a retained Llama control path for the main evidence tables, and no remaining “placeholder” status for `windowed_kv` or `hybrid_state` within the stated unsupported-feature boundary.
