# Appendix C. Correctness and Semantic Envelope

## C.1 Exactness Claims by Regime

| Regime | What is exact | What is not promised |
|---|---|---|
| `adaptive_full` | Full resident visibility, observer-only usage collection, greedy-token parity with the supported non-adaptive baseline under each **fully gated** configuration; Llama Family A remains the principal documented benchmark case | Partially gated configurations, multimodal paths, `compile_decode`, legacy `quantized_kv_start`, external cache reuse, or unsupported model types |
| `adaptive_soft` | Exact adaptive policy semantics, exact ordering of resident state, observer-only control, exact mixed-tier attention over the actual resident state | Greedy-token parity once compressed KV fidelity influences logits |
| `adaptive_hard` | Real replay-backed recovery, exact required-resident barrier before decode, honest pressure and live-byte accounting | Throughput parity with an all-resident baseline under small budgets |

## C.2 What “Observer-Only” Means Here

Adaptive KV does not modify:

- queries,
- keys,
- values,
- logits,
- or sampling policy.

The adaptive system observes attention participation and changes only memory residency. This is the key semantic boundary that makes the design safe to reason about: policy changes affect which representations are resident and when replay occurs, not the transformer equations themselves.

## C.3 Replay and Recovery Safety

The retained implementation enforces three important recovery invariants:

1. An evicted block cannot re-enter resident state by metadata change alone.
2. If decode requires history that is currently evicted, replay is performed before the forward that consumes that history.
3. Replay is anchored to authoritative source-token spans rather than synthetic reconstruction shortcuts.

This is why HARD behavior remains semantically safe even when throughput is well below the all-resident baseline.

## C.4 Resident-State Correctness

Resident state is constructed as an ordered concatenation of non-evicted segments. The key correctness properties are:

- logical token order is preserved,
- FULL and COMPRESSED segments are both visible to attention in the correct sequence,
- global masking and softmax operate over the assembled resident order,
- usage attribution is recorded in resident order and then mapped back to logical blocks.

The segmented executor may cost more than a fully fused path, but it does not change the semantics of attention over the resident state.

## C.5 Adaptive-Full Parity Restoration

The retained correctness story distinguishes between an actual control-path bug and compression-fidelity drift.

An earlier `adaptive_full` greedy mismatch on workload `C5` was traced to the usage-sampling path. The forward pass temporarily left the fused all-FULL attention path even though the resident state was contiguous and entirely full-fidelity. The retained resolution preserves the fused path for the forward output and derives usage information separately when needed.

The resulting retained interpretation is:

- `adaptive_full` is expected to match the non-adaptive greedy reference within the supported Llama baseline,
- and the focused `C5` reproductions checked into the repository no longer show a first difference on the retained prompt slices.

## C.6 Adaptive-Soft Drift Classification

Historical investigations also identified a different class of discrepancy: soft-only token drift under active compression. This matters because it has different implications.

The retained classification is:

- if drift appears only while compressed KV is resident,
- while `adaptive_full` remains aligned,
- then the first hypothesis should be compression-fidelity effects rather than a mixed-tier control bug.

The retained V1 scope therefore does **not** promise greedy parity for `adaptive_soft` under active compression, even though the design remains observer-only and semantically well-defined.

The focused `T4` reproductions currently checked into the repository do not show a first difference on the retained prompt slices, but this does not change the scope statement above. A no-difference reproduction is useful as a regression guard; it is not a guarantee that every future compressed-KV prompt will remain token-identical.

## C.7 Support Honesty

The retained platform architecture now exposes support explicitly through a structured capability model. This is important for correctness in the broad sense of system behavior:

- supported configurations are accepted and exercised,
- partial or unsupported configurations are rejected with explicit reasons,
- and the architecture does not imply broader working model support than the implementation actually provides.

At the time of this report, the implementation retains **three** exact runtime substrates (Families A, B, and C). The **principal parity and throughput evidence packaged here** remains the Llama full-attention Family A baseline; Family B (`ministral3`) and Family C (text-only `qwen3_5`) are retained exact paths where the capability gate reports full support, without expanding the unsupported-feature list.

## C.8 How to Read Noisy Source Material

The repository contains both:

- historical artifacts collected at intermediate implementation stages,
- and later retained regression checks.

Those materials are consistent at the level that matters most:

- semantic safety was progressively strengthened,
- replay-backed recovery is retained,
- the final repeated synthetic rerun reports `reference_token_match = true` across the regression matrix,
- and the retained implementation baseline is the platform architecture described in the main report.

Where individual historical artifacts differ in detail, this package resolves them by giving priority to:

1. the final retained architecture and freeze position,
2. repeated final synthetic reruns,
3. focused parity reproductions,
4. then earlier intermediate diagnostics only as evolution evidence.
