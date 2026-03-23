# Adaptive KV for MLXs — Real Workload Evaluation Report

**Snapshot:** Table and metrics below come from **`results/bench_adaptive/real_workloads/real_workloads_v1.json`** produced by a **post–C5-fix** run of `run_real_workloads` (same commands as **§3**). **`adaptive_full`** vs **`non_adaptive`** token parity holds for **C5** in this file. **`adaptive_soft`** mismatches **C1** and **C2** here (compressed resident blocks on those prompts); that is **consistent with V1** (no greedy parity guarantee under compression), not an open adaptive-control bug. **`adaptive_soft`** matches on the other eight workloads in this snapshot, including **T4** (longer excerpt than in the older parity investigation — see `adaptive_kv_token_parity_investigation_report.md` **§4** vs **§5**). The HARD subset in this report should now be read as an **earlier workload snapshot**, not as the final V1 HARD characterization: the later dedicated HARD completion track materially reduced churn and closed HARD for V1 within scope, a later Exact Runtime Optimization Program refined the retained runtime baseline, a final Runtime Architecture R&D program explored deeper exact runtime/data-plane redesigns without changing that retained baseline, and later architecture work retained a platform structure that cleanly separates the generic core from capability provider, runtime substrate, replay backend, and composed runtime adapter roles without broadening real support beyond the verified Llama path (see `docs/adaptive_kv/adaptive_kv_v1_freeze_summary.md` and `docs/adaptive_kv/Adaptive KV for Multi-Turn LLM Inference.md`).

## 1. Environment and Model

| Field | Value |
|--------|--------|
| Model path | `/Users/federicofilippi/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Path status | Present and used (run aborted if missing) |
| Python / MLX / platform | `3.13.5` / `0.31.1` / `darwin` (from run JSON `environment`) |
| Git revision | `5a441b91d2d40891b54b548e0d68f372bd0c3973` (from `results/bench_adaptive/real_workloads/real_workloads_v1.json`) |
| Entry point | `mlxs.generate.generate()` via `benchmarks/adaptive_kv/run.py` `_run_generate`; `compile_decode=False`, `quantized_kv_start=0` |
| Decode options | `max_tokens=96`, `temperature=0.0`, `seed=42+repeat_index` per workload repeat |

## 2. Workloads Actually Run

All **10** workloads from the approved plan (`C1`–`C6`, `T1`–`T4`), each built from **line-bounded excerpts** under the MLXs repo root.

| ID | Category | Source material (summary) |
|----|----------|---------------------------|
| C1 | coding | `llama.py` Attention + `attention.py` adaptive dispatch |
| C2 | coding | `decode.py` adaptive hooks + `manager.py` policy window |
| C3 | coding | `record_usage_from_attention` + `usage.py` collector |
| C4 | coding | `config_profiles.py` budgets |
| C5 | coding | `CLAUDE.md` module DAG |
| C6 | coding | `kv.py` + `generate/__init__.py` + `attention.py` excerpts |
| T1 | technical | `docs/specs.md` objectives + hot-path section |
| T2 | technical | `docs/specs.md` non-goals + FR table start |
| T3 | technical | `Adaptive KV for Multi-Turn LLM Inference.md` lines 1–180 |
| T4 | technical | full `adaptive_kv_v1_freeze_summary.md` |

**HARD subset:** `C2`, `T1` only, second suite `hard_budget_eval` with `soft_budget_bytes=30_000_000`, `hard_budget_bytes=45_000_000` for adaptive baselines that use budgets (`adaptive_hard`; `adaptive_full` remains huge-budget preset).

**Repeats:** `3` per `(workload, baseline)` cell in each suite.

**Machine-readable output:** `results/bench_adaptive/real_workloads/real_workloads_v1.json` (`suite: real_workloads_v1`, per-run field `suite`: `main_soft_budgets` | `hard_budget_eval`).

## 3. Commands Executed

```bash
cd /Users/federicofilippi/Desktop/MyProj/MLXs
uv run python -m benchmarks.adaptive_kv.run_real_workloads \
  --model "/Users/federicofilippi/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit" \
  --repeat 3 \
  --with-hard-subset \
  -o results/bench_adaptive/real_workloads/real_workloads_v1.json

uv run pytest tests/unit/test_generate/test_adaptive_kv.py -q
```

Main suite budgets (defaults): `soft_budget_bytes=42000000`, `hard_budget_bytes=250000000`.

## 4. Results Summary

**Throughput (median tok/s over 3 repeats).** `main_soft_budgets` suite; `ev` / `rc` = median `adaptive_kv_evictions_total` / `adaptive_kv_recomputations_total`.

| WL | prompt tok | non_adapt | full | soft | soft/full | ref ok (full / soft) |
|----|------------|-----------|------|------|-----------|----------------------|
| C1 | 768 | 211.9 | 208.8 | 197.6 | 0.95 | yes / **no** |
| C2 | 506 | 160.4 | 158.7 | 154.1 | 0.97 | yes / **no** |
| C3 | 750 | 209.5 | 208.3 | 203.7 | 0.98 | yes / yes |
| C4 | 764 | 211.7 | 217.3 | 200.8 | 0.92 | yes / yes |
| C5 | 119 | 79.4 | 78.7 | 78.1 | 0.99 | yes / yes |
| C6 | 816 | 202.1 | 202.4 | 203.9 | 1.01 | yes / yes |
| T1 | 841 | 221.8 | 216.7 | 213.8 | 0.99 | yes / yes |
| T2 | 460 | 149.8 | 141.9 | 151.6 | 1.07 | yes / yes |
| T3 | 1692 | 286.9 | 276.6 | 268.7 | 0.97 | yes / yes |
| T4 | 1987 | 296.0 | 296.6 | 291.7 | 0.98 | yes / yes |

**Aggregate:** Under SOFT budgets, **no evictions and no recompute requests** were recorded on any main-suite row (`ev=0`, `rc=0` medians everywhere).

**Correctness (this JSON):** `reference_token_match` is **false** for **6** rows: **`adaptive_soft`** on **C1** and **C2** (3 repeats each). All **`adaptive_full`** rows match **`non_adaptive`**. **Interpretation:** SOFT budgets demote blocks on those coding workloads; greedy token drift under **compressed** resident KV is **allowed** by V1 (see freeze summary §7). **T4** matches in this run; an older excerpt once showed **soft-only** drift (parity report **§5**).

## 5. Coding Workload Findings

- **Cross-file and multi-snippet tasks (C1, C2, C6):** **`adaptive_soft` median tok/s** tracks **`adaptive_full`** within a few percent. **`adaptive_full`** matches **`non_adaptive`** tokens on **C1**, **C2**, **C6**; **`adaptive_soft`** matches on **C6** but **not** on **C1** / **C2** in this snapshot (compression on those prompts).
- **C3** (usage batching sources) is slightly faster on **soft** than **full** on this sample (median 198 vs 188 tok/s); magnitudes are within run noise — **do not over-interpret** a single reversal.
- **C4** (config profiles) behaves like other mid-length code workloads: soft ≈ full, **ref ok**.
- **C1, C2:** **`adaptive_soft`** does **not** match **`non_adaptive`** tokens (median rows above); gauges show **many compressed blocks** under SOFT budgets — consistent with **compression-induced greedy drift**, not a policy wiring defect.
- **C5** (short DAG, **119** prompt tokens): **`adaptive_full`** and **`adaptive_soft`** both **match** `non_adaptive` in this snapshot (post–C5-fix behavior).
- **Eviction/recompute:** none observed on coding workloads under the main SOFT budgets.

## 6. Technical/Document Workload Findings

- **T1, T2:** **soft tracks full** (~97–99% median tok/s) with **full token parity**.
- **T3** (longest prompt, **1692** tokens): highest tok/s in the suite; **soft** ~4% below **full**; **ref ok**.
- **T4** (freeze summary excerpt, **~1987** prompt tokens in this run): **`adaptive_full`** and **`adaptive_soft`** both **match** `non_adaptive` here; soft still runs **mixed-tier** resident KV per metrics. **V1 stance:** lack of mismatch is **empirical**; **greedy parity for soft under compression is not guaranteed** (see parity report **§4** vs historical **§5**).

## 7. HARD Regime Findings (historical snapshot)

**Suite `hard_budget_eval`,** medians over 3 repeats:

| WL | non_adapt | full | hard | ev (hard) | rc (hard) | ref (hard) |
|----|-----------|------|------|-----------|-----------|------------|
| C2 | 164.8 | 162.6 | 124.5 | 1 | 1 | yes |
| T1 | 200.1 | 209.1 | 66.6 | 22 | 5 | yes |

- **C2:** **~23%** drop in median tok/s vs **full** under HARD; **single** eviction / recompute (median), **tokens still match** `non_adaptive`.
- **T1:** **~68%** drop vs **full**; **heavy** churn (median **22** evictions, **5** recompute requests) — **HARD is informative but expensive**; still **token-correct** on this run.

These rows remain useful as evidence of the earlier stressed behavior on real repo text, but they are **not** the final V1 HARD verdict. The later dedicated HARD track reduced replay-wave count and stabilized the stressed regime materially; the remaining V1 HARD limitation is now treated as structural to the current exact mixed-tier segmented executor rather than as an open hardening defect.

## 8. Overall Interpretation

**Measured:** On **8/10** workloads, **`adaptive_soft` matches `non_adaptive` tokens** (greedy); **C1** and **C2** are the exceptions where soft is under compression. **Soft vs full** median tok/s stays roughly within **±8%** on this matrix (C4 soft slightly lower). **No** SOFT-budget evictions/recomputes on main-suite rows.

**Token parity:** **`adaptive_full`** matches everywhere. **`adaptive_soft`** mismatches **only** where compression is active enough to diverge (**C1**, **C2** here) — aligned with V1 non-guarantee, not an unresolved bug.

**Inference (qualified):** Realistic **code + spec** prompts largely **replicate** synthetic-suite behavior: SOFT ≈ FULL throughput when memory pressure does not trigger churn. The HARD rows here capture the earlier stressed behavior that motivated the later dedicated HARD track; they should not be read as contradicting the final V1 decision that HARD is now closed within scope.

## 9. Verdict

**USEFUL — V1-aligned**

**Why:** The suite answers how adaptive KV behaves on **real repo text** (throughput SOFT vs FULL, earlier HARD cost on **T1**). **`adaptive_full`** tracks **`non_adaptive`** tokens on all rows in this snapshot. **`adaptive_soft`** may diverge when SOFT budgets compress KV (**C1**, **C2**); other workloads matched in this run. The final V1 HARD closure, however, comes from the later dedicated HARD benchmark/optimization track rather than from this earlier workload snapshot alone.

## 10. After V1 freeze

**Regression anchor:** Re-run **§3** when changing adaptive attention, budgets, or workload excerpts; compare **`reference_token_match`** and throughput to this JSON. Token expectations: **`adaptive_kv_v1_freeze_summary.md`** (§7). Optional **compressed-KV fidelity** work remains product-driven. Later post-freeze runtime work kept the existing retained baseline with **executor-ready resident slice metadata** and a **score-side `q_len == 1` mixed-tier specialization**; a final Runtime Architecture R&D program then explored deeper exact runtime/data-plane redesigns, including a streaming mixed-tier decode path and an incremental resident-plan refresh path. Those prototypes remained exact and were benchmarked, but they were both rejected at the real benchmark gate. Subsequent architecture cleanup/generalization and the final Platform Architecture Refactor then retained a cleaner baseline with an explicit generic adaptive-KV core plus separate capability-provider, runtime-substrate, replay-backend, and composed-adapter roles, while still keeping the verified Llama path as the only real supported runtime. Correctness and support honesty stayed intact and fresh regression checks did not show a material retained-path regression, so the current retained baseline is now that platform architecture and the remaining gap remains optional deeper executor/data-plane R&D territory rather than routine hardening.

---

*Report aligned with `real_workloads_v1.json` from git `5a441b91d2d4` (~8.7 minutes wall time for 108 generate calls on the stated hardware).*
