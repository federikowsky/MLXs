MLXs — Phase 0 Execution Checklist

Status: execution artifact
Role: operational checklist for Phase 0 of the MLXs refactor

⸻

1. Purpose of Phase 0

Phase 0 exists to unblock safe execution of the refactor by freezing the operational starting conditions before any Layer 1 code work begins.

It must close, in an explicit and verifiable way:
	•	branch strategy,
	•	runtime subtree carve-out,
	•	retained vs replaced code areas,
	•	temporary shim assumptions,
	•	initial execution discipline.

Phase 0 exists so that Phase 1 starts with:
	•	no ambiguity about what is being rebuilt,
	•	no ambiguity about what is being preserved,
	•	no ambiguity about which old contracts are no longer canonical.

⸻

2. Phase 0 goals

Complete all of the following:
	•	Freeze branch/workspace strategy.
	•	Freeze the runtime-core replacement boundary.
	•	Freeze retained non-runtime substrate boundaries.
	•	Confirm current repo coupling hotspots against the canonical docs.
	•	Confirm reuse/discard classification at operational level.
	•	Freeze temporary shim policy.
	•	Protect the benchmark path from accidental breakage before Phase 2.
	•	Produce a Phase 1 handoff state with no unresolved baseline ambiguity.

⸻

3. Branch and workspace setup checklist

Mark each item complete before touching the runtime core.
	•	Create or designate the refactor execution branch as the canonical working branch for the rebuild.
	•	Confirm that the current mainline remains the reference branch for comparison, not the place for direct runtime-core mutation.
	•	Freeze the architectural document pack as the governing source for execution:
	•	Foundations
	•	Research Dossier
	•	Benchmark Protocol
	•	Repo Baseline Audit
	•	Performance Core Spec
	•	General Path Spec
	•	Advanced Engines Spec
	•	Product Surfaces Spec
	•	Implementation Plan
	•	Record the exact repo baseline revision from which execution begins.
	•	Record the exact non-runtime substrate baseline chosen for retention.
	•	Record the exact runtime subtree areas designated for replacement.
	•	Confirm that no new architectural decisions are allowed to enter Phase 1 without explicit review against the canonical docs.
	•	Confirm that no “refactor in place” of the old universal generate/decode center will be treated as the default strategy.

⸻

4. Runtime carve-out checklist

Classify current repo areas into replace, retain, or uncertain pending verification.

4.1 Mark as “to replace”
	•	Current canonical generate/decode center.
	•	Old broad generate() boundary as architectural root.
	•	Old mixed runtime/semantics/orchestration logic in generation paths.
	•	Compile-centered runtime integration that shaped the old runtime contracts.
	•	Any runtime path that still encodes “universal engine” assumptions.

4.2 Mark as “retained substrate”
	•	Model loading and registry infrastructure, pending interface conformance checks.
	•	Model architecture implementations.
	•	Shared layers and model substrate.
	•	Cache substrate implementations as components, not as proof of current runtime contracts.
	•	Converter/model-format code.
	•	Broad non-runtime utilities.
	•	Stable config/observability infrastructure outside hot-path contracts.

4.3 Mark as “retain with extraction/refactor”
	•	Chunked prefill logic.
	•	Useful benchmark harness assets and benchmark corpus/reporting assets.
	•	Narrow eager-first ideas from the late branch, if they can be extracted without preserving contaminated boundaries.
	•	Selected server shell pieces only if reattached above new lower-layer contracts.
	•	Selected cache abstractions only if narrowed to fit new layer boundaries.

4.4 Mark as “uncertain pending verification”
	•	Prompt-cache integration points.
	•	Existing batching subsystem coupling to old generate contracts.
	•	Existing speculative subsystem coupling to old generate contracts.
	•	Existing config-to-runtime mapping surfaces that may encode old boundary assumptions.
	•	Existing public runtime API surfaces that may need temporary shims.

⸻

5. Boundary confirmation checklist

Confirm the current repo against the target layer model.

5.1 Layer 1 boundary confirmation
	•	Identify all files/modules currently acting as the effective runtime core.
	•	Confirm where active generation state is currently owned.
	•	Confirm where active cache mutation is currently owned.
	•	Confirm where eval/materialization decisions are currently made.
	•	Confirm all current points where product or semantics concerns enter the hot path.

5.2 Layer 2 boundary confirmation
	•	Identify where single-request rich generation semantics currently live.
	•	Confirm where stop semantics are currently handled.
	•	Confirm where logprobs / penalties / processors are currently shaped.
	•	Confirm whether these concerns currently leak into the core.

5.3 Layer 3 boundary confirmation
	•	Identify where batching currently depends on old runtime contracts.
	•	Identify where speculative decoding currently depends on old runtime contracts.
	•	Identify where prompt-cache orchestration currently depends on old runtime contracts.
	•	Confirm all current multi-request coordination hotspots.

5.4 Layer 4 boundary confirmation
	•	Identify current server/CLI/config/observability/lifecycle surfaces.
	•	Confirm where these surfaces currently shape lower-layer behavior incorrectly.
	•	Confirm current composition-root locations.
	•	Confirm whether any lower layer currently behaves like a product-shaped composition root.

5.5 Critical coupling confirmation
	•	Confirm generate ↔ server coupling hotspots.
	•	Confirm generate ↔ prompt-cache coupling hotspots.
	•	Confirm generate ↔ batch coupling hotspots.
	•	Confirm generate ↔ compile coupling hotspots.
	•	Confirm generate ↔ config coupling hotspots.

⸻

6. Reuse / discard confirmation checklist

Use the Repo Baseline Audit categories operationally.

6.1 Keep
	•	Confirm retained model-loading substrate.
	•	Confirm retained model/layer substrate.
	•	Confirm retained cache substrate components.
	•	Confirm retained converter/model-format components.
	•	Confirm retained non-runtime utilities.

6.2 Keep with extraction
	•	Confirm whether chunked prefill can be reused directly or only after extraction.
	•	Confirm whether eager-first narrow-slice ideas can be reused without importing old contracts.
	•	Confirm benchmark harness components safe to preserve.
	•	Confirm whether any server shell pieces are safe to preserve above new boundaries.

6.3 Rework heavily
	•	Mark current public generation entry surfaces as non-canonical.
	•	Mark broad decode-loop logic as subject to heavy rework or replacement.
	•	Mark compile-related generation integration as subject to heavy rework.
	•	Mark prompt-cache integration around generate contracts as subject to heavy rework.
	•	Mark batching/speculative integration points as subject to heavy rework if coupled to old runtime boundaries.

6.4 Explicitly exclude from the new core
	•	Old universal generate/decode contract.
	•	Old compile-centered runtime assumptions.
	•	Old resident-state assumptions.
	•	Old product-shaped runtime metadata requirements.
	•	Old broad context/recipe/session objects crossing hot-path boundaries.

⸻

7. Temporary shim policy checklist

7.1 Where shims are allowed
	•	Layer 4 compatibility surfaces during later reattachment.
	•	Transitional bridges from old product surfaces to new lower-layer contracts.
	•	Temporary adaptation between old upper-layer expectations and new Layer 1 / Layer 2 contracts.

7.2 Where shims are not allowed
	•	Inside the new Performance Core.
	•	As hidden widening of Layer 1 contracts.
	•	As a way to keep the old universal generate boundary alive.
	•	As hidden orchestration inside Layer 2.
	•	As product-shaping inside Layers 1–3.

7.3 Shim labeling rules

Every shim must be:
	•	explicitly marked as transitional,
	•	scoped to a specific phase boundary,
	•	assigned a removal condition,
	•	barred from becoming the canonical contract by convenience.

⸻

8. Benchmark readiness checklist for later Phase 2

Prepare now what must not be broken before benchmark alignment.
	•	Preserve benchmark harness assets needed for canonical Class A benchmarking.
	•	Preserve or snapshot current comparable mlx-lm benchmark setup assumptions.
	•	Preserve benchmark input corpora / prompt targets / decode targets currently used as reference material.
	•	Record current benchmark environment assumptions that must remain reproducible.
	•	Confirm that Phase 1 runtime-core replacement will not destroy the ability to run the canonical fast-path benchmark later.
	•	Confirm that diagnostics or instrumentation added in Phase 1 will be separable from primary measurement runs.
	•	Confirm that compile/no-compile, stream policy, and cache-clearing behaviors remain observable/configurable enough for later Phase 2 benchmarking.
	•	Confirm that no temporary shim planned for Phase 1 will pollute the canonical benchmark path.

⸻

9. Phase 0 exit criteria

Phase 0 is complete only when all of the following are true:
	•	The execution branch and baseline revision are fixed.
	•	The runtime-core replacement boundary is fixed.
	•	The retained non-runtime substrate boundary is fixed.
	•	The current repo coupling hotspots have been confirmed operationally.
	•	Reuse / extraction / rework / discard categories have been confirmed at working level.
	•	Temporary shim policy is frozen and documented.
	•	Benchmark-readiness preservation requirements are frozen.
	•	There is no ambiguity about which old contracts are no longer canonical.
	•	There is no ambiguity about what Phase 1 is allowed to rebuild.
	•	There is no ambiguity about what Phase 1 is not allowed to preserve.

Phase 0 is not complete if any of the above remains unresolved.

⸻

10. Immediate handoff to Phase 1

Phase 1 may start only when the following are true:
	•	The new Performance Core subtree target is unambiguous.
	•	The old generate/decode center is formally demoted from canonical status.
	•	The Layer 1 contract from the Performance Core Spec is the only valid reference for the new core.
	•	No unresolved decision remains about whether to “refactor in place” versus replace the runtime core subtree.
	•	Retained substrate components needed by Layer 1 are identified.
	•	Excluded legacy structures are explicitly blacklisted from the new core.
	•	Benchmark-path preservation constraints are known to the Phase 1 executor.
	•	Temporary shims are understood as non-core, future-removal structures only.

Nothing critical for Layer 1 may remain ambiguous at handoff.