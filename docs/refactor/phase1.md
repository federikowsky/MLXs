MLXs — Phase 1 Execution Checklist

Status: execution artifact
Role: operational checklist for Phase 1 of the MLXs refactor

⸻

1. Purpose of Phase 1

Phase 1 exists to produce the new Layer 1 / Performance Core subtree as a real, separate runtime foundation.

It must make two things true:
	•	the new Performance Core exists as the only canonical Layer 1 boundary;
	•	the old generate/decode center is no longer the canonical runtime root.

Phase 1 is complete only when the refactor has a real sovereign core subtree, not a renamed or cleaned-up version of the previous universal generation path.

⸻

2. Phase 1 goals
	•	Build the new Layer 1 subtree as a distinct runtime center.
	•	Establish the Layer 1 boundary in code, not only in documents.
	•	Make core input/output/state boundaries explicit from the start.
	•	Establish singular ownership of active mutable generation state.
	•	Establish singular ownership of active cache state.
	•	Preserve explicit eval/sync/materialization discipline.
	•	Preserve compatibility with later benchmark alignment without polluting the core with benchmark logic.
	•	Prevent old runtime contracts from re-entering Layer 1.

⸻

3. Layer 1 subtree creation checklist

3.1 Boundary-first creation
	•	Create or isolate a new runtime-core subtree as a distinct code area.
	•	Ensure the subtree is structurally identifiable as Layer 1 and not embedded inside old public generation entrypoints.
	•	Ensure Layer 1 can be reasoned about without traversing product-surface or orchestration code.

3.2 Minimum areas that must exist immediately
	•	A core runtime boundary for prefill.
	•	A core runtime boundary for decode-step progression.
	•	A Layer 1-visible cache ownership boundary.
	•	A Layer 1-visible execution-boundary policy surface.
	•	A Layer 1-visible minimal token-selection contract.
	•	A Layer 1-local state model boundary.

3.3 Areas that must be isolated or carved out first
	•	Separate Layer 1 runtime code from old broad generate() entry logic.
	•	Separate Layer 1 from Layer 2 semantic shaping concerns.
	•	Separate Layer 1 from Layer 3 orchestration concerns.
	•	Separate Layer 1 from Layer 4 transport/config/observability concerns.

3.4 Areas of old runtime that must not be treated as root
	•	Old universal generate() boundary.
	•	Old mixed decode loop that carries semantics/orchestration/product concerns together.
	•	Old compile-centered runtime shaping as architectural root.
	•	Old runtime wrappers whose main function was to keep one universal engine alive.

⸻

4. Core contract checklist

4.1 Input boundary
	•	Confirm Layer 1 input contract is narrow and execution-relevant only.
	•	Confirm Layer 1 does not require product-facing request objects.
	•	Confirm Layer 1 does not require orchestration/session-management objects from higher layers.
	•	Confirm Layer 1 does not require broad options objects spanning multiple layers.
	•	Confirm only generation-ready runtime inputs reach Layer 1.

4.2 Output boundary
	•	Confirm Layer 1 output contract is minimal.
	•	Confirm Layer 1 outputs do not include product-shaped events.
	•	Confirm Layer 1 outputs do not require broad semantic metadata by default.
	•	Confirm optional richer outputs are not forced into the canonical Layer 1 boundary.

4.3 State boundary
	•	Confirm Layer 1 has an explicit active runtime-state boundary.
	•	Confirm the boundary excludes product/session/orchestration state not required for token advancement.
	•	Confirm Layer 1 does not carry broad request context through the runtime path.

4.4 Cache ownership
	•	Confirm Layer 1 is the sole owner of active mutable cache state during generation.
	•	Confirm no adjacent layer mirrors active mutable cache state.
	•	Confirm cache mutation semantics are defined inside Layer 1, not delegated upward.

4.5 Eval / sync / materialization discipline
	•	Confirm Layer 1 defines explicit evaluation boundaries.
	•	Confirm Layer 1 does not rely on accidental host materialization.
	•	Confirm scalar extraction and host conversion are visible and controlled.
	•	Confirm sync points are enumerable and attributable to the Layer 1 contract.

⸻

5. Runtime ownership checklist

5.1 Mutable state ownership
	•	Confirm one active owner of mutable generation runtime state exists during a Layer 1 execution flow.
	•	Confirm no product-facing or orchestration-facing structures hold parallel active mutable runtime state.
	•	Confirm no old “context” or “recipe” object is acting as hidden owner of Layer 1 state.

5.2 Cache ownership
	•	Confirm Layer 1 owns active cache lifecycle during active generation.
	•	Confirm upper layers may observe/export only through explicit boundaries.
	•	Confirm Layer 1 does not depend on upper-layer policy to mutate active cache state correctly.

5.3 Explicit prohibitions
	•	No mirror structures for active mutable runtime state inside or adjacent to Layer 1.
	•	No per-step export/import cycles as part of the core path.
	•	No adapter chain translating equivalent runtime state structures inside the core.
	•	No deep-copy-based state handoff inside canonical Layer 1 execution flow.

⸻

6. Execution-discipline checklist

6.1 Eval boundaries
	•	Confirm all allowed eval boundaries in Layer 1 are explicit in structure and intent.
	•	Confirm no debug or incidental eval path exists in the core subtree.
	•	Confirm eval placement is part of runtime design, not left implicit.

6.2 item() / materialization rules
	•	Confirm all scalar extraction points are explicit.
	•	Confirm no array printing in Layer 1 runtime paths.
	•	Confirm no NumPy conversion in Layer 1 runtime paths.
	•	Confirm host-side materialization is not hidden in helpers/wrappers.

6.3 Sync explicitness
	•	Confirm synchronization is explicit and attributable.
	•	Confirm no hidden sync enters via logging, debugging, wrappers, or broad helper objects.
	•	Confirm sync behavior is visible enough to support later benchmark alignment.

6.4 Stream-discipline assumptions to preserve
	•	Confirm Layer 1 preserves explicit stream ownership assumptions rather than relying on product-layer behavior.
	•	Confirm Layer 1 does not embed assumptions that would make later stream-policy refinement impossible.
	•	Confirm Layer 1 does not mix stream concerns with product or orchestration state.
	•	Confirm Layer 1 keeps RNG/stream relations preservable for future higher-layer use without embedding Layer 2/3 policy.

⸻

7. Legacy exclusion checklist

The following patterns must be explicitly excluded from the new Layer 1.

7.1 Universal generate boundary
	•	Do not use the old broad generate() contract as the Layer 1 root.
	•	Do not preserve old all-in-one generation entry surfaces as canonical Layer 1 boundaries.

7.2 Compile-first shaping
	•	Do not shape the new core around compile assumptions.
	•	Do not let future compile compatibility widen current Layer 1 contracts.
	•	Do not preserve old compile-centric state contracts in Layer 1 by inertia.

7.3 Broad contexts / recipes
	•	Exclude broad recipe/context/session objects from the Layer 1 hot path.
	•	Exclude aggregated cross-layer option carriers from the canonical Layer 1 contract.

7.4 Product-shaped outputs
	•	Exclude product-ready event objects from Layer 1 outputs.
	•	Exclude transport-facing metadata from Layer 1 outputs.
	•	Exclude server/CLI compatibility shaping from Layer 1.

7.5 Old contamination patterns
	•	Exclude semantics/orchestration/product logic from the core runtime subtree.
	•	Exclude old resident-state assumptions from shaping Layer 1.
	•	Exclude old prompt-cache orchestration assumptions from shaping Layer 1.

⸻

8. Allowed reuse checklist

8.1 Components that may enter Layer 1 directly
	•	Reuse of model/layer substrate only as lower-level compute substrate, not as proof of old runtime contracts.
	•	Reuse of cache substrate components where compatible with Layer 1 ownership rules.
	•	Reuse of chunked prefill logic only if it conforms to the new Layer 1 boundaries.

8.2 Components allowed only with extraction
	•	Narrow eager-first ideas from prior work only if extracted from contaminated public runtime boundaries.
	•	Selected helper utilities only if stripped of product/semantics/orchestration assumptions.
	•	Selected diagnostics hooks only if they are outside canonical runtime execution and removable from hot paths.

8.3 Components that must not enter the new core
	•	Old broad public generate wrappers.
	•	Old universal decode loop structures.
	•	Old runtime logic carrying Layer 2 semantics inline.
	•	Old runtime logic carrying Layer 3 orchestration inline.
	•	Old Layer 4 transport/config/observability shaping in or adjacent to the core contract.

⸻

9. Phase 1 validation checklist

9.1 Validate that Layer 1 exists as a true separate subtree
	•	A distinct Layer 1 code area exists.
	•	Layer 1 can be identified without traversing Layer 2–4 code.
	•	Layer 1 is not merely a thin wrapper around old generate/decode code.
	•	Layer 1 has its own explicit runtime boundary.

9.2 Validate conformance to the Performance Core Spec
	•	Input boundary matches Layer 1 contract discipline.
	•	Output boundary is minimal.
	•	State ownership is singular and visible.
	•	Cache ownership is singular and visible.
	•	Eval/sync/materialization rules are explicit.
	•	Product semantics are absent from Layer 1.
	•	Orchestration concerns are absent from Layer 1.

9.3 Validate that the old core is no longer canonical
	•	Old broad generate/decode center is not the new runtime root.
	•	No new lower-layer code depends on the old universal generate boundary as architecture.
	•	Any temporary reference to old paths is explicitly transitional and outside the new canonical Layer 1.
	•	The new Layer 1 is the only valid reference for future Phase 2 benchmark alignment.

9.4 Validate no hidden recontamination
	•	No broad options object has crept into Layer 1.
	•	No Layer 2 semantic object has become mandatory in Layer 1.
	•	No Layer 3 orchestration state has become mandatory in Layer 1.
	•	No Layer 4 product-facing object has become mandatory in Layer 1.

⸻

10. Phase 1 exit criteria

Phase 1 is complete only when all of the following are true:
	•	The new Layer 1 / Performance Core subtree exists as a distinct runtime center.
	•	The Layer 1 boundary is explicit in code.
	•	Input/output/state/cache ownership boundaries are explicit and conformant.
	•	Eval/sync/materialization discipline is explicit and visible.
	•	The old universal generate/decode center is no longer the canonical runtime boundary.
	•	Legacy-excluded patterns have been kept out of the new core.
	•	Allowed reused components, if present, do not reintroduce old contracts.
	•	The subtree is ready to be measured under the Benchmark Protocol in Phase 2.
	•	No unresolved ambiguity remains about what Layer 1 is.

Phase 1 is not complete if any of the above remains unresolved.

⸻

11. Immediate handoff to Phase 2

Phase 2 may begin only when the following are true:
	•	The new Layer 1 boundary is stable enough to benchmark.
	•	Canonical fast-path execution can be isolated from legacy product semantics.
	•	Benchmark-path preservation constraints from Phase 0 remain intact.
	•	Eval/materialization/sync boundaries are visible enough to support benchmark alignment.
	•	Stream-policy assumptions in Layer 1 are explicit enough for benchmark labeling and fairness normalization.
	•	No temporary shim is embedded inside the canonical Layer 1 benchmark path.
	•	The old core is not still being measured by accident instead of the new Layer 1.

Nothing necessary for benchmark alignment and validation may remain ambiguous at handoff.