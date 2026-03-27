# Adaptive KV benchmark harness

Single-request benchmarks for **adaptive KV v1** using the real [`generate()`](../../src/mlxs/generate/__init__.py) entry point. Compatibility is decided by `assess_generation_compatibility`; the **documented** synthetic and real-workload tables in [`docs/adaptive_kv_report/`](../../docs/adaptive_kv_report/) use **Llama** full-attention. Retained **ministral3** (Family B) and text-only **qwen3_5** (Family C) runs are supported when the gate passes—use `--preflight` before large sweeps.

## Requirements

- Install the package in editable mode so `mlxs` is importable:

  ```bash
  pip install -e ".[dev]"
  ```

- Run with the **repository root** on `PYTHONPATH` so the `benchmarks` package resolves:

  ```bash
  cd /path/to/MLXs
  PYTHONPATH=. python -m benchmarks.adaptive_kv --help
  ```

## Preflight (compatibility)

Loads the model and prints JSON from `assess_generation_compatibility` (exit code 1 if unsupported):

```bash
PYTHONPATH=. python -m benchmarks.adaptive_kv --preflight --model /path/to/Llama
```

## Canonical entry points

- `python -m benchmarks.adaptive_kv.run`: main synthetic benchmark harness.
- `python -m benchmarks.adaptive_kv.run_real_workloads`: retained real-workload suite.
- `python -m benchmarks.adaptive_kv.investigate_token_parity`: focused parity reproduction for the checked-in C5/T4 investigation.
- `python -m benchmarks.adaptive_kv.compare`: compare two stored JSON runs.

Older one-off microbench / profiler scripts are intentionally not part of the retained benchmark surface.

## Full run

Writes machine-readable JSON (and optional Markdown summary):

```bash
PYTHONPATH=. python -m benchmarks.adaptive_kv \
  --model /path/to/Llama \
  --scenario smoke long_static \
  --baseline non_adaptive adaptive_full adaptive_soft adaptive_hard \
  --budget-profile default \
  --soft-budget-bytes 80000000 \
  --hard-budget-bytes 400000000 \
  -o results/adaptive_kv.json \
  --summary-md results/adaptive_kv.md
```

For tiny models / stress similar to unit tests, use `--budget-profile tiny_stress`.

## Compare two JSON outputs

```bash
PYTHONPATH=. python -m benchmarks.adaptive_kv.compare run_a.json run_b.json
```

## Notes

- `adaptive_kv_recomputations_total` counts **recompute requests**, not replayed token count (see `notes` in the JSON output).
- `peak_rss_bytes_rusage` uses `getrusage`; on macOS `ru_maxrss` is bytes.
- Set explicit `--soft-budget-bytes` / `--hard-budget-bytes` for your model after inspecting `final_resident_bytes` from an `adaptive_full` run.
