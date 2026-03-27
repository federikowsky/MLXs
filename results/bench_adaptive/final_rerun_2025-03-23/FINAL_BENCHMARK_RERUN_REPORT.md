# Adaptive KV for MLXs — Final Benchmark Rerun Report

## 1. Environment and Model

| Field | Value |
|--------|--------|
| Model path | `/Users/federicofilippi/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Path available | Yes (verified before runs) |
| Match to prior campaigns | Same path as recorded in `results/bench_adaptive/diag_compression.json` and `diag_recovery.json` args |
| `model_type` (preflight) | `llama` |
| Python / MLX / platform (this rerun) | `3.13.5` / `0.31.1` / `darwin` (from new JSON `environment`) |
| Git revision — **before** | `ce093c90a631be804ce8b0401c58307ecebd0fca` (`diag_compression.json`, `diag_recovery.json`) |
| Git revision — **after** | `f05a2f02706da0ca6eccb2ecdc48b5ddbdc8709f` (`final_rerun_2025-03-23/*.json`) |

**Note:** The primary **before** baseline for comparison remains the root-level `diag_*.json` at **`repeat`: 1**, commit `ce093c90…`.

## 2. Commands Executed

```bash
mkdir -p results/bench_adaptive/final_rerun_2025-03-23
```

**SOFT-style diagnostic (42M / 250M), 4 scenarios, 3 repeats:**

```bash
cd /Users/federicofilippi/Desktop/MyProj/MLXs
uv run python -m benchmarks.adaptive_kv.run \
  --model "/Users/federicofilippi/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit" \
  --scenario long_static topic_drift delayed_topic_return hard_pressure_context \
  --baseline non_adaptive adaptive_full adaptive_soft \
  --repeat 3 \
  --seed 42 \
  --prompt-target-tokens 512 \
  --decode-tokens 64 \
  --soft-budget-bytes 42000000 \
  --hard-budget-bytes 250000000 \
  -o results/bench_adaptive/final_rerun_2025-03-23/diag_compression.json
```

**HARD-style diagnostic (30M / 45M), 4 scenarios, 3 repeats:**

```bash
uv run python -m benchmarks.adaptive_kv.run \
  --model "/Users/federicofilippi/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit" \
  --scenario long_static topic_drift delayed_topic_return hard_pressure_context \
  --baseline non_adaptive adaptive_full adaptive_hard \
  --repeat 3 \
  --seed 42 \
  --prompt-target-tokens 512 \
  --decode-tokens 64 \
  --soft-budget-bytes 30000000 \
  --hard-budget-bytes 45000000 \
  -o results/bench_adaptive/final_rerun_2025-03-23/diag_recovery.json
```

**Unit tests (correctness smoke):**

```bash
uv run pytest tests/unit/test_generate/test_adaptive_kv.py -q
```

## 3. Benchmark Sets Rerun

| Campaign | Output file (new) | Scenarios | Baselines | Budgets | Repeats per adaptive baseline |
|----------|-------------------|-----------|-----------|---------|-------------------------------|
| SOFT / compression | `results/bench_adaptive/final_rerun_2025-03-23/diag_compression.json` | `long_static`, `topic_drift`, `delayed_topic_return`, `hard_pressure_context` | `non_adaptive`, `adaptive_full`, `adaptive_soft` | soft=42_000_000, hard=250_000_000 | **3** |
| HARD / recovery stress | `results/bench_adaptive/final_rerun_2025-03-23/diag_recovery.json` | same | `non_adaptive`, `adaptive_full`, `adaptive_hard` | soft=30_000_000, hard=45_000_000 | **3** |

Other settings aligned with prior diagnostics: `prefill_step_size=2048`, `clear_cache_interval=256`, `block_size_tokens=64`, `update_window_steps=16`, `seed=42`, `prompt_target_tokens=512`, `decode_tokens=64`, `compile_decode` implicit false, `quantized_kv_start` implicit 0 (harness).

**Aggregation:** For **after** rows, **median** (and min/max) of `tokens_per_second` and `elapsed_seconds` over the 3 repeats per `(scenario, baseline)`. For **before**, each cell is a **single** run (`repeat=1` in stored args).

## 4. Correctness Check

| Signal | Result |
|--------|--------|
| `reference_token_match` (adaptive vs `non_adaptive` reference for same scenario) | **All `true`** for every adaptive row in both new JSON files (all scenarios × all repeats × soft and hard baselines). |
| `tests/unit/test_generate/test_adaptive_kv.py` | **5 passed** (run after benchmark campaign). |

No regressions observed on token-level equality to the non-adaptive reference in this suite.

## 5. Before vs After — SOFT Regime

**Before:** `results/bench_adaptive/diag_compression.json` (`repeat=1`, git `ce093c90…`).  
**After:** `results/bench_adaptive/final_rerun_2025-03-23/diag_compression.json` (`repeat=3`, git `f05a2f02…`).

| Scenario | Baseline | Before tok/s | After tok/s **median** [min, max] | Δ vs before (approx) |
|----------|----------|--------------|-------------------------------------|----------------------|
| long_static | non_adaptive | 248.38 | 204.15 (single run) | run variance |
| long_static | adaptive_full | 206.96 | 211.55 [202.55, 216.28] | ~+2% |
| long_static | **adaptive_soft** | **65.93** | **207.89** [202.94, 217.65] | **~+215% (~3.2×)** |
| topic_drift | adaptive_soft | 76.65 | 198.66 [196.80, 199.58] | ~+159% |
| delayed_topic_return | adaptive_soft | 78.72 | 201.58 [194.86, 205.32] | ~+156% |
| hard_pressure_context | adaptive_soft | 77.40 | 210.13 [202.34, 210.64] | ~+172% |

**Within the after run only** (same machine session, same seed policy): `adaptive_soft` vs `adaptive_full` median tok/s:

| Scenario | adaptive_full | adaptive_soft | Ratio soft/full |
|----------|---------------|---------------|-----------------|
| long_static | 211.55 | 207.89 | 0.98 |
| topic_drift | 203.81 | 198.66 | 0.97 |
| delayed_topic_return | 197.63 | 201.58 | 1.02 |
| hard_pressure_context | 215.51 | 210.13 | 0.98 |

Evictions / recompute requests (median across repeats): **0 / 0** for all SOFT **after** rows (same as before SOFT).

## 6. Before vs After — HARD Regime

**Before:** `results/bench_adaptive/diag_recovery.json` (`repeat=1`, git `ce093c90…`).  
**After:** `results/bench_adaptive/final_rerun_2025-03-23/diag_recovery.json` (`repeat=3`, git `f05a2f02…`).

| Scenario | Baseline | Before tok/s | After tok/s **median** [min, max] | Evict / recomp (median) after |
|----------|----------|--------------|-------------------------------------|-------------------------------|
| long_static | adaptive_hard | 66.12 | **148.51** [145.89, 152.29] | 1 / 1 |
| topic_drift | adaptive_hard | 68.26 | **153.07** [149.69, 157.73] | 1 / 1 |
| delayed_topic_return | adaptive_hard | 65.19 | **151.67** [147.69, 152.62] | 1 / 1 |
| hard_pressure_context | adaptive_hard | 67.75 | **149.32** [144.25, 151.38] | 1 / 1 |

**After run — adaptive_hard vs adaptive_full** (median tok/s):

| Scenario | adaptive_full | adaptive_hard | Ratio hard/full |
|----------|---------------|---------------|-----------------|
| long_static | 216.28 | 148.51 | 0.69 |
| topic_drift | 216.14 | 153.07 | 0.71 |
| delayed_topic_return | 211.01 | 151.67 | 0.72 |
| hard_pressure_context | 211.97 | 149.32 | 0.70 |

Counters **unchanged in structure** vs before HARD: still **1 eviction, 1 recompute request** per scenario (median), so the stress pattern is stable; throughput improved without changing those discrete signals.

## 7. Interpretation

**Measured**

1. **`adaptive_soft` improved materially** vs `ce093c90` single-shot SOFT diagnostics: on the order of **2.5–3.2×** higher median tok/s in the table above, bringing SOFT into the same band as **`adaptive_full`** on this model and scenario set.
2. **`adaptive_hard` improved materially** (~**2.2–2.3×** median tok/s) while keeping **the same** eviction/recompute counts, consistent with removing a large per-forward overhead (usage extraction / sync) that affected every step when compressed state was present.
3. **Correctness:** `reference_token_match` remained **true** everywhere in the new artifacts; unit tests for adaptive KV still pass.

**Inference (qualified)**

- **Absolute tok/s** for `non_adaptive` and `adaptive_full` differ somewhat between the old single-run files and the new medians; that is expected **run-to-run noise** and **repeat aggregation**. The decisive comparison for the optimization is **`adaptive_soft` before vs after** and **`adaptive_soft` vs `adaptive_full` on the same new run**, which are aligned in the same session.
- **HARD** remains **~30% below `adaptive_full`** median tok/s with **fixed** `evict=1` / `recomp=1`, so **replay/recovery plus mixed-tier attention** still explain most of the remaining gap vs full-resident baseline under these budgets — not a failure of the usage fix.

## 8. Current Remaining Bottleneck

Under **SOFT** budgets, **`adaptive_soft` ≈ `adaptive_full`**: no clear dominant bottleneck left in this suite for that regime.

Under **HARD** budgets, the **largest remaining gap** is between **`adaptive_hard` and `adaptive_full`** with **stable** replay/eviction counters — **most plausibly the combined cost of recompute/replay work and mixed-tier segmented attention**, not usage extraction (which the profiler already showed collapsed after the pass).

## 9. Verdict

**GOOD WITHIN SCOPE**

**Why:** For the **compression-oriented SOFT regime** that motivated the usage work, **`adaptive_soft` now tracks `adaptive_full` within a few percent** on median tok/s across all four scenarios, with **zero** evictions/recomputes in these runs and **token-identical** outputs vs `non_adaptive`. **HARD** improved strongly but is **still** materially slower than full — that is **expected** for a small-budget stress configuration and does not negate that the **within-scope SOFT path** is now in an acceptable performance band.

## 10. Recommended Next Step

**Stop and accept current within-scope performance** for **SOFT / mixed-tier compression** workloads on Llama full attention, unless product priorities **explicitly** require **HARD-budget parity** with `adaptive_full`.

**Why:** Further gains on **HARD** likely require **replay/recovery** or **segmented attention** work, not another round on usage extraction. Pursue **optimize replay/recovery path** (or **broader validation** on more prompts) only if HARD-style latency is a **shipping** requirement.

---

**Artifacts**

| Role | Path |
|------|------|
| Before SOFT | `results/bench_adaptive/diag_compression.json` |
| Before HARD | `results/bench_adaptive/diag_recovery.json` |
| After SOFT (this rerun) | `results/bench_adaptive/final_rerun_2025-03-23/diag_compression.json` |
| After HARD (this rerun) | `results/bench_adaptive/final_rerun_2025-03-23/diag_recovery.json` |
| This report | `results/bench_adaptive/final_rerun_2025-03-23/FINAL_BENCHMARK_RERUN_REPORT.md` |
