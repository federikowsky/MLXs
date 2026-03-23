# Adaptive KV V1 — Freeze summary

**Status:** Frozen V1 engineering handoff summary.  
**Normative requirements** remain in `docs/specs.md`; this file is a **standalone snapshot** of scope, behavior, architecture, and validated outcomes (no dependency on checked-in benchmark artifacts).

**Related docs (V1):** `docs/dev/adaptive_kv_token_parity_investigation_report.md` (C5 root cause + T4 classification + refreshed `parity_*.json`), `docs/dev/adaptive_kv_real_workload_evaluation_report.md` (table derived from `results/bench_adaptive/real_workloads/real_workloads_v1.json`), `docs/Adaptive KV for Multi-Turn LLM Inference.md` (paper-length freeze narrative).

---

## 1. Scope frozen (approved)

Single-request generation only, **`model_type == "llama"`**, full-attention Llama baseline, homogeneous `list[KVCache]` baseline, block-level policy with tiers **FULL / COMPRESSED / EVICTED**, real replay/recovery, observer-only semantics for usage, unsupported families rejected at compatibility check.

**Explicitly out of this freeze:** `compile_decode`, legacy `quantized_kv_start` stress as a product path, external cache reuse, non-Llama architectures, batched/server composition.

---

## 2. Behavioral contract (correctness)

- **`adaptive_full`** (all resident blocks FULL, no compression in the attention path): for a fixed scenario, seed, and generation options, token sequences must match the **`non_adaptive`** reference on the regression harness (greedy / `temperature=0`). A prior parity gap on real workload **C5** was traced to usage sampling forcing the forward off fused SDPA; the implementation now keeps **`mx.fast.scaled_dot_product_attention`** for the attention output when resident state is **all-FULL contiguous** and computes usage weights via a separate matmul/softmax path only when sampling is required (see `src/mlxs/layers/attention.py`, `_is_all_full_contiguous_resident_state`).
- **`adaptive_soft`**: when the policy holds **COMPRESSED** resident KV, **greedy token-by-token parity with `non_adaptive` is not guaranteed**. Quantized resident attention is a deliberate fidelity tradeoff; observed mismatches (e.g. real workload **T4**) align with **compression-induced quality drift**, not with a mixed-tier wiring bug (shadow dequantized cache reproduces the same flip — useful as a future diagnostic).
- **`adaptive_hard`**: **semantically correct** (replay-backed recovery, correct resident visibility) and now treated as **good enough under stress within the approved V1 scope**; **throughput and memory behavior** are not targeted for parity with **`adaptive_full`**, and the remaining gap is structural to the current exact segmented executor.
- Unit gate: `tests/unit/test_generate/test_adaptive_kv.py` (5 tests) passes.

---

## 3. Hot-path flow (reference)

1. **`decode_loop`** (`src/mlxs/generate/decode.py`): `before_decode_forward` → `ensure_required_resident` (replay if needed) → model forward → `after_decode_forward`.
2. **Attention** (`src/mlxs/layers/attention.py`): mixed-tier or non-contiguous FULL layout uses segmented SDPA; **all-FULL contiguous** resident state uses fused SDPA for the **output** and, when usage must be sampled, derives **weights** from a separate full-precision scores path so policy signals stay available without swapping the main kernel away from baseline numerics. The retained post-freeze runtime baseline also keeps **executor-ready resident slice metadata** plus a **score-side decode specialization** for **`q_len == 1`** mixed-tier resident state, while preserving the generic segmented executor as the exact fallback/oracle.
3. **Usage collection** (`src/mlxs/adaptive_kv/manager.py`, `src/mlxs/adaptive_kv/usage.py`): per-layer forward enqueues block-level usage scalars via **`record_batch`** + **`mx.async_eval`**; **`snapshot_and_reset`** on policy windows performs batched **`mx.eval`** and host reduction, replacing the previous per-step sync pattern that dominated wall time on mixed-tier paths.

---

## 4. Performance outcomes (frozen characterization)

Validation used a **small Llama 3-class instruct model** (4-bit MLX bundle, ~3B parameters), **Apple Silicon**, **Python 3.13** and **MLX 0.31.x** class stack. Figures are **order-of-magnitude and relative**; re-run the project benchmark CLI to reproduce absolute tok/s on your machine.

| Regime | Budgets (soft / hard bytes) | Outcome |
|--------|-----------------------------|---------|
| **SOFT** | 42_000_000 / 250_000_000 | After usage batching, **`adaptive_soft` throughput approaches `adaptive_full`** (within a few percent on median over repeated runs) on fixed scenarios (`long_static`, `topic_drift`, `delayed_topic_return`, `hard_pressure_context`). Versus the pre-optimization implementation, **`adaptive_soft` improved on the order of ~3×** on the same matrix. A later same-machine exact runtime program kept a narrow executor/runtime subset and measured **+1.6%** on `long_static` and **+0.6%** on `hard_pressure_context`, without changing SOFT semantics. |
| **HARD** | 30_000_000 / 45_000_000 | After the dedicated HARD passes and a later exact runtime program, **`adaptive_hard` is considered good enough under stress for V1**. The retained runtime baseline keeps **executor-ready resident slice metadata**, a **score-side `q_len == 1` mixed-tier decode specialization**, and the **generic executor as exact fallback**. A later deeper structural executor redesign was also explored and validated, but the decisive same-process 1024-token HARD gate improved by only about **~1.6%**, below the intended retention threshold, so that redesign was reverted. The current retained baseline therefore remains the earlier retained implementation, and the remaining gap is still attributed to the current exact mixed-tier segmented executor rather than to an obvious local overhead bug. |

**Profiler lesson (mixed-tier, coalesced compressed segment):** before usage batching, **wall time inside usage extraction could exceed ~70%** of end-to-end time in a narrow micro-configuration; after the change, that path is **no longer the dominant share** in the SOFT diagnostic regime, and SOFT **tracks full-resident adaptive** closely.

---

## 5. Engineering verdict

- **Within SOFT / compression scope:** **GOOD** — performance is **acceptable relative to adaptive full** on the frozen diagnostic matrix; **`adaptive_full` greedy parity vs `non_adaptive`** is **expected** after the fused-SDP guard for all-FULL contiguous state.
- **`adaptive_soft` token parity:** **not promised** under active compression; treat divergences as **expected quality loss** unless the product defines bitwise greedy parity through quantized KV (V1 does not).
- **HARD stress scope:** Improved materially and now **closed for V1**. The later Exact Runtime Optimization Program was executed seriously, validated, and is now **closed** with the existing retained baseline still in place; the deeper structural executor redesign was **not** retained because the real flagship 1024-token HARD gain was too small to justify the added complexity. **Remaining gap vs full** is structural to the current exact mixed-tier segmented executor, not a sign of unfinished routine hardening. Further work here is **optional executor R&D**, not normal V1 completion work.

**Default next action:** treat adaptive KV V1 as **feature-frozen** for the approved Llama single-request SOFT and HARD scope unless product requires higher compressed-KV fidelity, a materially smaller HARD gap, or broader support. If HARD is reopened, do it as **dedicated executor R&D**, not as ordinary hardening.

---

## 6. Revalidation (how to reproduce locally)

```bash
cd /path/to/MLXs
uv run pytest tests/unit/test_generate/test_adaptive_kv.py -q
uv run python -m benchmarks.adaptive_kv.run \
  --model "<local-llama-mlx-model-path>" \
  --scenario long_static topic_drift delayed_topic_return hard_pressure_context \
  --baseline non_adaptive adaptive_full adaptive_soft \
  --repeat 3 --seed 42 \
  --prompt-target-tokens 512 --decode-tokens 64 \
  --soft-budget-bytes 42000000 --hard-budget-bytes 250000000
```

Repeat with `--baseline non_adaptive adaptive_full adaptive_hard` and `--soft-budget-bytes 30000000 --hard-budget-bytes 45000000` for the HARD regime. Add `-o <path>` if you want JSON on disk; paths and history are **your** bookkeeping, not part of this freeze doc.

---

## 7. Guarantees vs non-guarantees (V1)

| Topic | Guaranteed in V1 scope | Not guaranteed |
|--------|-------------------------|----------------|
| **`adaptive_full` vs `non_adaptive`** | Greedy token parity when attention uses the same fused path as baseline for all-FULL contiguous resident KV (regression + unit tests). | Parity if future changes break the all-FULL fused path or introduce non-equivalent numerics. |
| **`adaptive_soft` vs `non_adaptive`** | Correct residency, mixed-tier attention, policy behavior per config. | Bitwise greedy token parity when COMPRESSED blocks participate (compression-induced drift). |
| **`adaptive_hard`** | Correctness after eviction (replay), safe failure modes. | Tok/s parity with **`adaptive_full`**; low overhead under heavy churn. |

---

## 8. Non-goals (do not scope-creep from this freeze)

- Training, multi-request batching, new model families without compatibility assessment.
- Changing tier semantics or removing replay without a new spec revision.
- Continuing HARD-regime optimization or the Exact Runtime Optimization Program as ordinary V1 hardening; any further gain now belongs to explicit executor R&D tied to product targets, with the current retained baseline otherwise left in place.
- Demanding **`adaptive_soft`** greedy parity with full-precision KV without a dedicated **compressed-KV fidelity** program.

---

*End of freeze summary. Update this file only when scope, spec, or validation intent changes deliberately.*
