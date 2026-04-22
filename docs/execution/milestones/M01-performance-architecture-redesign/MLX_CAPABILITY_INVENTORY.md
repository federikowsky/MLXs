# M01.R1b MLX Capability Inventory

Updated: 2026-04-20

## Phase 0 — Framing

- Purpose: pressure-test the completed `M01.R1` research against the wider official MLX capability surface.
- Non-goal: this does not replace the completed MLXs/`mlx_lm`/MLX architecture research.
- Decision rule: if a wider MLX capability family clearly displaced the current thesis, update the thesis; otherwise resume `M01.R2`.

## Phase 1 — Official-source inventory

Primary sources:

- MLX docs
  - Lazy Evaluation
  - Compilation
  - Using Streams
  - Transforms
  - Unified Memory
  - Export Functions
  - Distributed Communication
  - `mlx.core.quantize`
  - `mlx.core.quantized_matmul`
  - `mlx.core.block_masked_mm`
  - Python ops index
- Installed `mlx.core` runtime/docstrings
- Installed `mlx_lm/generate.py`

## Phase 2 — Capability classification

| Capability family | What it is | MLXs uses it now | `mlx_lm` uses it now if known | Relevance | Leverage | Constraints / risks | Influence on thesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lazy evaluation | graph builds lazily; unused outputs still build graphs | yes | yes | relevant now | high | graph-build cost persists even when outputs are dropped | reinforces |
| `eval` / sync | explicit evaluation boundary | yes | yes | relevant now | high | fixed eval overhead and graph-size overhead | reinforces |
| `async_eval` | async trigger for arrays/trees | yes in benchmark/helper and some runtime paths | yes in single-request and batch generation | relevant now | high | experimental API; must stay bounded | reinforces |
| Streams | per-op stream/device routing | yes | yes | relevant now | high | stream ownership must stay explicit | reinforces |
| Compile / shapeless | compiled graph transform and shape-sensitive reuse | yes | not in comparable eager path | relevant now | medium | recompilation on shape/type/input-count change; cannot replace wrong ownership | reinforces as secondary |
| Unified memory | CPU/GPU share one memory pool; scheduler manages dependencies across streams | implicit | implicit | relevant now | medium | helps overlap decisions but does not remove hot-loop structural taxes | no change |
| Memory controls | `clear_cache`, peak/cache memory, wired limit, device info | yes | yes | relevant now | medium | important for stability/fairness, not main decode tax | no change |
| Export/import | export/import MLX functions and traces | no | no known use | relevant later | low | shape/type-bound, experimental, not hot-path | no change |
| Distributed communication | `distributed.*` collectives and point-to-point | no | no known current hot-path use | likely irrelevant now | low | outside current single-host acceptance scope | no change |
| Quantization ops | `quantize`, `dequantize`, `quantized_matmul` | yes | yes indirectly | relevant later | medium | useful for model/cache internals, not primary single-request split | no change |
| NN quantization helpers | `mlx.nn.quantize`, quantized modules | yes | unknown | relevant later | low | tooling/model-format oriented, not current redesign driver | no change |
| Selection primitives | `topk`, `argpartition`, `argsort`, `take`, `take_along_axis` | yes for richer/logprob paths | unknown direct hot-path use | relevant later | low-to-medium | not main minimal-decode bottleneck | no change |
| Specialized matmul/gather/block ops | `gather_mm`, `gather_qmm`, `block_masked_mm`, `quantized_matmul` | yes in model layers | not visible in `generate.py` | relevant later | medium | likely model-layer/MoE leverage, not the current benchmark/core split | no change |
| File/model I/O | `load`, `save_safetensors`, `save_gguf` | yes | yes | likely irrelevant now | low | not hot-path | no change |

## Phase 3 — Addendum result

### Relevant now

- lazy evaluation
- `eval`
- `async_eval`
- streams
- compile constraints
- unified memory as a constraint/opportunity
- memory controls

### Relevant later

- quantization ops
- specialized matmul/gather/block ops
- selection primitives for enriched paths
- export/import

### Likely irrelevant under current project constraints

- distributed communication
- file export/import as a redesign driver

### Thesis impact

- No change to the ranked redesign thesis.

Why:

- the dominant current tax is still the benchmark/core split and broader loop ownership fragmentation
- no newly inventoried MLX capability provides a stronger first move than core-first resident progression unification
- the fallback remains batch-first resident progression redesign

## Consequence

- Resume `M01.R2 first bounded redesign slice — runtime_core.run_greedy lookahead-path adoption`

## Phase 4 — R29 consequence

- Current local installed versions:
  - `mlx 0.31.1`
  - `mlx-lm 0.31.2`
- Current official MLX surface rechecked:
  - latest MLX release is still `v0.31.1`
  - official docs still expose:
    - streams
    - custom extensions
    - custom Metal kernels
- R29 result:
  - these capabilities remain available in the design space
  - they are not justified for the current `AC1` front because the accepted-sensitive local substrate probe did not isolate one dominant substrate-level bottleneck
- Consequence:
  - carry the capability surface forward to the next strongest front
  - do not treat `AC1` as a built-in-MLX or custom-kernel problem by default on current evidence

## Phase 5 — R30 consequence

- Local `R30` lower-boundary readback:
  - current decisive staggered surface still reproduces the late-request starvation gap
  - below that, the measurable split is eval-boundary / owner-pipeline cost rather than attention, mask, or cache-update kernels
- Capability consequence:
  - MLX streams and `async_eval` remain relevant ingredients
  - they are not justified as a standalone path on the current `AC2` evidence because they appear as part of a broader generation-owner contract difference
  - custom extensions and custom Metal kernels are not justified on the current `AC2` evidence because no dominant op-level bottleneck was isolated

## Phase 6 — first integrated `R30` candidate consequence

- Rejected local candidate result:
  - repaired the staggered late-request step positions
  - still lost badly on the real staggered surface because it duplicated generation-owner work
- Capability consequence:
  - this failure still points at internal batch ownership, not at missing MLX primitives
  - built-in MLX and custom-kernel paths remain out
