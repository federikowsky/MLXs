MLXs — Phase 0 Operational Closeout

Status: execution artifact
Role: records the completed Phase 0 freeze and the exact Phase 1 handoff state
Date: 2026-04-10

---

1. Freeze record

- Execution branch: `refactor/core-exec`
- Clean execution worktree: `/Users/federicofilippi/Desktop/MyProj/MLXs-core-exec`
- Tracked-code baseline: `2e2ce77f7ffd60622a9b7c9b2f6e69fcf594620f`
- Governing pack snapshot commit: `20b9a79`
- Mainline remains comparison-only during the refactor.
- The dirty `refactor/core` worktree is not the Phase 1 execution workspace.

---

2. Phase 0 checklist completion

- Branch/workspace strategy: complete
- Runtime-core replacement boundary: complete
- Retained non-runtime substrate boundary: complete
- Coupling-hotspot confirmation: complete
- Reuse/discard classification: complete
- Temporary shim policy: complete
- Benchmark-path preservation freeze: complete

Phase 0 is operationally closed on this branch.

---

3. Repo classification

3.1 Replace

- `src/mlxs/generate/__init__.py`
- `src/mlxs/generate/decode.py`
- `src/mlxs/generate/compile.py`
- `src/mlxs/protocols/generate.py` as the architectural root

3.2 Retain

- `src/mlxs/load/**`
- `src/mlxs/models/**`
- `src/mlxs/layers/**`
- `src/mlxs/cache/**` as substrate components
- `src/mlxs/convert/**`
- tokenization, model-loading support, and non-hot-path observability/config substrate

3.3 Retain with extraction/refactor

- `src/mlxs/generate/prefill.py`
- `src/mlxs/generate/stop.py`
- `src/mlxs/generate/logits.py`
- `src/mlxs/generate/sampling.py`
- `benchmarks/mlxs_vs_mlx_lm/**`
- selected server transport shells above new lower-layer boundaries
- eager-first donor history from commit `6b56b62`, treated as donor history only

3.4 Uncertain pending later verification

- prompt-cache integration around `src/mlxs/server/chat.py` and `src/mlxs/prompt_cache/lru.py`
- `src/mlxs/batch/**`
- `src/mlxs/speculative/**`
- config-to-runtime mapping surfaces in `src/mlxs/config/schema.py` and `src/mlxs/server/deps.py`

---

4. Coupling hotspots

- `generate ↔ server`: direct wiring through `src/mlxs/server/deps.py`, `src/mlxs/server/routes/chat.py`, and `src/mlxs/server/chat.py`
- `generate ↔ prompt-cache`: cache import/export and deep-copy prompt-cache handoff
- `generate ↔ batch`: duplicated prefill/decode semantics in `src/mlxs/batch/scheduler.py`
- `generate ↔ speculative`: reuse of generation-layer prefill/sampling/stop in `src/mlxs/speculative/verify.py`
- `generate ↔ compile`: compiled step closure over mutable cache in `src/mlxs/generate/compile.py`
- `generate ↔ config`: broad runtime and semantic knobs mixed into `GenerateConfig`

---

5. Runtime carve-out boundary

Phase 1 creates a fresh Layer 1 subtree at:

- `src/mlxs/runtime_core/`

Allowed inside Layer 1:

- prefill
- decode-step progression
- active cache ownership
- explicit eval/sync policy
- minimal token-selection result
- core-local runtime state

Explicitly excluded from Layer 1:

- current `mlxs.generate` public root
- `TokenEvent`
- broad `GenerateOptions`
- stop/logprob/penalty shaping
- prompt-cache policy and cache export/import shaping
- compile-first shaping
- server/config/batch/speculative/product objects

---

6. Old contracts no longer canonical

The following remain legacy compatibility surfaces only and must not define the new core:

- `mlxs.generate.generate(...) -> Iterator[TokenEvent]`
- `GenerateProtocol` as the core contract
- `TokenEvent` as the default core output
- `GenerateOptions` as the Layer 1 input carrier
- `StopCondition` in the hot path
- in-loop logprob/top-logprob shaping
- in-loop repetition-penalty shaping
- compile-centered runtime shaping via `compile_decode`, `forward_fn`, and warmup
- prompt-cache handoff through `cache=` and `final_cache_out`

---

7. Shim policy

- Transitional shims are allowed only above Layer 1.
- No shim may live inside `src/mlxs/runtime_core/`.
- Current `mlxs.generate.generate` may become a compatibility wrapper later, but it is not the canonical runtime boundary.
- Any shim must be explicitly transitional and removable during later phases.

---

8. Benchmark preservation

- Canonical benchmark assets remain `benchmarks/mlxs_vs_mlx_lm/**`.
- The current local benchmark edits in the dirty `refactor/core` worktree are not the canonical Phase 0 baseline.
- Phase 1 must preserve the ability to benchmark the new Layer 1 without product-surface contamination.
- Benchmark-harness changes, if needed, are a later explicit step and not part of the Phase 0 freeze.

---

9. Phase 1 handoff

Phase 1 starts from this branch/worktree only.

Binding handoff rules:

- Build Layer 1 as a fresh subtree at `src/mlxs/runtime_core/`.
- Do not refactor the old `src/mlxs/generate/` tree in place and call it Layer 1.
- Reuse only substrate and extracted donors that conform to Layer 1 ownership and boundary rules.
- Keep Layer 2 semantics, Layer 3 orchestration, and Layer 4 transport/product shaping out of the new core.
- Keep the benchmark path isolated and reproducible for Phase 2.

Phase 1 may proceed.
