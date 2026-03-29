# Archival decision record — Exact Adaptive KV execution-pack track

**Status:** Accepted (track archived)  
**Scope:** Engineering and product documentation only. This record does not change runtime code.

## Context

MLXs carried a multi-phase program to build an **exact adaptive KV** serving architecture: family-aware runtime integration, replay/recovery-aware behavior, explicit pressure management, redesign of cadence/dormancy/wake, and a retained substrate based on **persistent exact execution slabs** with **execution packs** and pack-native streaming attention. Resident outcomes are exposed as `TQ_SAFE`, `TQ_AGGR`, and `EVICTED` in configuration and types.

The work produced a **frozen retained baseline** that is strong on control-plane clarity, correctness within the support envelope, and operational behavior under stress.

## Decision

The **exact adaptive execution-pack track** is **archived** as the active mainline strategy for closing adaptive-vs-baseline throughput in comfortable and long regimes. The retained implementation stays in the tree as a **valid technical result**. Mainline effort **pivots** to a **pragmatic** `FULL` / `COMPRESSED` / `EVICTED` direction (high-level intent only; detailed spec is a follow-on).

## What was achieved

- A **retained exact adaptive serving architecture**: policy external to attention semantics, family-aware adapters, honest capability gating, explicit replay-backed recovery, pressure-managed operation.
- **Execution-pack baseline** above exact slabs, with oracle validation for supported Family A / B / C combinations on the matrices described in the canonical technical specification.
- **Product-level evidence** consistent with treating this path as **operational serving control** (evictions, recomputes, bounded behavior under tight budgets), not as a demonstrated universal improvement in user-visible answer quality or comfortable-regime throughput vs ordinary baseline inference.

## What was not closed

- **Throughput:** A retainable closure vs the normal non-adaptive baseline in **comfortable / long** regimes was **not** achieved.
- **Residual locus:** Evidence progressively isolated the remaining gap to the **exact adaptive attention compute path on split (multi-)pack** resident views, rather than to unresolved generic repository architecture elsewhere.

## Why stop this line on the main track

- Further meaningful progress on the residual gap points toward **lower-level compute, backend, or operator** work that is **outside** the pragmatic scope chosen for MLXs at this stage.
- Multiple **documented closure attempts** (data-plane ideas, dual-lane designs, dormant-path mirrors, dense-prefix caches for split-pack compute, executor/sync redesigns as “final” answers, Family C local passes, cold-path churn work past diminishing returns, etc.) were **tried and rejected on evidence**. Continuing the same track without a new class of backend capability would mostly repeat explored ground.

This is a **scope and priority decision**, not a verdict that the retained baseline is worthless.

## Rejected directions (archival index)

The canonical specification (`adaptive_kv_turboquant_first.spec.md`) §16 lists historical and structural rejections (lossy TurboQuant resident path, transient grouped-segment hot path, harmful view-query optimizations, etc.). The same specification adds an archival index of **throughput- and substrate-closure attempts** that did not yield a retainable win. Result artifacts under `results/bench_adaptive/` remain the empirical record where present.

## Positioning of the frozen baseline

- **Valid:** Exact, family-gated, pressure-managed adaptive serving within the stated support matrix.
- **Not claimed:** Default universal choice for best throughput in comfortable regimes; “TurboQuant-first” as a **lossy compression centerpiece**; or “almost closed” wording for the comfortable/long throughput gap.

## Next direction (follow-on program)

**Intentional pivot:** Pragmatic **`FULL` / `COMPRESSED` / `EVICTED`** tiering with:

- **`FULL`** — baseline-like default where memory allows.
- **`COMPRESSED`** — a simple, genuinely useful compressed resident tier.
- **`EVICTED`** — last-resort eviction with explicit recovery semantics as appropriate.

**Emphasis:** Simplicity, throughput, and pragmatic runtime behavior rather than universal exact-adaptive idealism.

This ADR does **not** specify schemas, migrations, or APIs for that program. The technical freeze for the execution-pack era remains described in `adaptive_kv_turboquant_first.spec.md`.

## References

- Canonical technical freeze: [`adaptive_kv_turboquant_first.spec.md`](./adaptive_kv_turboquant_first.spec.md)
- Repo integration map: [`dev/repo_mapping_kv_cache.md`](./dev/repo_mapping_kv_cache.md)
- Historical V1 report package (`FULL` / `COMPRESSED` / `EVICTED` evaluation era): [`adaptive_kv_report/`](./adaptive_kv_report/)
