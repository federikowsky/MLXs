MLXs — Phase 6 Execution Checklist

Status: execution artifact
Role: operational checklist for Phase 6 of the MLXs refactor

⸻

1. Purpose of Phase 6

Phase 6 exists to close the refactor as an integrated system rather than as a set of individually completed layer implementations.

It must close:
	•	integration hardening,
	•	cleanup of transitional structures,
	•	compatibility reduction,
	•	removal of residual legacy contamination,
	•	final verification that the implemented system matches the canonical 4-layer architecture in practice, not only on paper.

Phase 6 comes after reintroduction of Layers 1–4 because cleanup and completion discipline are meaningful only once the full target architecture exists end-to-end.

⸻

2. Phase 6 goals
	•	Identify and reduce or remove temporary shims, adapters, bridges, and compatibility layers.
	•	Eliminate remaining traces of the old universal generate/decode center.
	•	Verify that Layers 1–4 exist as real operational layers with canonical boundaries.
	•	Verify that no upper layer has widened or redefined a lower layer during integration.
	•	Preserve benchmark separation and cost attribution after full integration.
	•	Confirm that the selected baseline strategy was actually realized:
	•	stable non-runtime substrate retained,
	•	fresh runtime-core subtree replacement completed.
	•	Reduce ambiguity in the integrated system so that canonical paths are obvious and transitional paths are minimized or removed.
	•	Produce a final go/no-go determination on architectural completion and benchmark-valid completion.

⸻

3. Transitional-structure cleanup checklist

3.1 Identify transitional structures still present
	•	Enumerate all temporary shims still present between old and new boundaries.
	•	Enumerate all temporary adapters between old product surfaces and new Layers 1–3.
	•	Enumerate all compatibility bridges introduced during phased reattachment.
	•	Enumerate all legacy-path fallback structures still reachable in normal execution.
	•	Enumerate all temporary benchmark-preservation structures still present.

3.2 Classify each transitional structure

For each structure, confirm:
	•	its original purpose,
	•	the phase that required it,
	•	whether that phase dependency still exists,
	•	whether it is still on any canonical execution path,
	•	whether it is only on a compatibility path.

3.3 Criteria to keep temporarily

A transitional structure may remain only if all are true:
	•	it still protects a real unresolved integration dependency,
	•	removing it now would break a canonical layer boundary or product surface without a ready replacement,
	•	it is explicitly labeled transitional,
	•	it has a clear post-Phase-6 removal condition.

3.4 Criteria to reduce or remove

A transitional structure must be reduced or removed if any is true:
	•	its original integration role is complete,
	•	it shadows a canonical lower-layer contract,
	•	it keeps a legacy boundary alive in practice,
	•	it introduces ambiguity about which path is canonical,
	•	it contaminates benchmark interpretation,
	•	it widens a lower-layer contract for compatibility convenience.

⸻

4. Legacy contamination cleanup checklist

4.1 Identify residues of the old universal generate/decode center
	•	Search for any remaining code paths where one broad entrypoint still effectively owns runtime + semantics + orchestration + product behavior.
	•	Identify any module still acting as the practical center of generation behavior across multiple layers.
	•	Confirm whether any old public generate boundary still functions as the de facto canonical path.

4.2 Identify legacy boundaries surviving under new names
	•	Check for old broad context/recipe/session carriers reintroduced under renamed abstractions.
	•	Check for old compile-first assumptions surviving in runtime-layer design.
	•	Check for old product-shaped runtime outputs surviving in Layer 1 or Layer 2.
	•	Check for old orchestration-aware runtime contracts surviving in Layer 1 or Layer 2.
	•	Check for old server-shaped generation boundaries surviving behind Layer 4.

4.3 Eliminate contamination residue in Layer 1
	•	Remove or quarantine any Layer 2, Layer 3, or Layer 4 concerns still present in Layer 1.
	•	Remove any old compatibility widening of Layer 1 input/output/state contracts.
	•	Remove any remaining hidden ownership duplication in active runtime/cache state.

4.4 Eliminate contamination residue in Layer 2
	•	Remove any Layer 3 orchestration concerns still present in Layer 2.
	•	Remove any Layer 4 transport/API/config shaping still present in Layer 2.
	•	Remove any old broad-generate assumptions still shaping Layer 2 boundaries.

4.5 Eliminate contamination residue in Layer 3
	•	Remove any Layer 1 execution-discipline logic that migrated upward incorrectly.
	•	Remove any Layer 2 semantic definitions that migrated into Layer 3.
	•	Remove any Layer 4 product-surface shaping still present in Layer 3.

4.6 Eliminate contamination residue in Layer 4
	•	Remove any hidden lower-layer ownership absorbed into Layer 4.
	•	Remove any old runtime-root behavior hiding inside server/CLI/product wiring.
	•	Remove any compatibility logic that effectively governs lower-layer architecture.

⸻

5. Architecture-conformance checklist

5.1 Verify that Layers 1–4 exist as real distinct layers
	•	Layer 1 exists as a distinct runtime-core subtree.
	•	Layer 2 exists as a distinct rich single-request semantics layer.
	•	Layer 3 exists as a distinct advanced orchestration layer.
	•	Layer 4 exists as a distinct product-surface layer.

5.2 Verify canonical boundaries per layer
	•	Layer 1 owns execution discipline, active runtime ownership, active cache ownership, and minimal token-generation contract.
	•	Layer 2 owns rich single-request generation semantics only.
	•	Layer 3 owns batching/speculative/prompt-cache orchestration and engine-local coordination only.
	•	Layer 4 owns server/CLI/config/observability/lifecycle/compatibility surfaces only.

5.3 Verify boundary enforcement in practice
	•	No canonical Layer 1 path requires Layer 2 logic to exist.
	•	No canonical Layer 2 path requires Layer 3 orchestration to exist.
	•	No canonical Layer 3 path requires Layer 4 transport shaping to exist.
	•	Layer 4 consumes lower layers rather than redefining them.

5.4 Verify Layer 4 is the only composition root
	•	Confirm no hidden composition root survives in Layer 1.
	•	Confirm no hidden composition root survives in Layer 2.
	•	Confirm no hidden composition root survives in Layer 3.
	•	Confirm all product-surface wiring happens only in Layer 4.

⸻

6. Contract-integrity checklist

6.1 Layer 1 contract integrity
	•	Confirm Layer 1 input contract has not widened during integration.
	•	Confirm Layer 1 output contract remains minimal.
	•	Confirm Layer 1 state ownership remains singular and explicit.
	•	Confirm Layer 1 cache ownership remains singular and explicit.
	•	Confirm no upper-layer metadata or policy is now mandatory in Layer 1.

6.2 Layer 2 contract integrity
	•	Confirm Layer 2 remains focused on rich single-request semantics.
	•	Confirm Layer 2 did not absorb Layer 3 orchestration or Layer 4 product shaping.
	•	Confirm Layer 2 still consumes Layer 1 without redefining it.

6.3 Layer 3 contract integrity
	•	Confirm Layer 3 remains focused on orchestration and coordination.
	•	Confirm Layer 3 did not absorb Layer 1 execution discipline.
	•	Confirm Layer 3 did not absorb Layer 2 semantic ownership.
	•	Confirm Layer 3 did not absorb Layer 4 transport/config/API concerns.

6.4 Layer 4 contract integrity
	•	Confirm Layer 4 remains focused on product surfaces and composition root behavior.
	•	Confirm Layer 4 did not redefine lower-layer contracts.
	•	Confirm Layer 4 did not absorb lower-layer runtime or orchestration ownership.

6.5 Cross-layer integrity
	•	Confirm no upper layer has widened a lower-layer canonical contract for convenience.
	•	Confirm no lower layer now depends on a higher-layer concept to function canonically.
	•	Confirm no compatibility shim has become a hidden contract-expansion mechanism.

⸻

7. Benchmark and cost-integrity checklist

7.1 Preserve Class A purity
	•	Confirm Class A remains a pure Layer 1 benchmark.
	•	Confirm no Layer 2 enrichment is silently on the Class A path.
	•	Confirm no Layer 3 orchestration is silently on the Class A path.
	•	Confirm no Layer 4 transport/config/observability overhead is silently on the Class A path.

7.2 Preserve Layer 2 benchmark separation
	•	Confirm Layer 2 benchmarks remain separate from Layer 1/Class A.
	•	Confirm Layer 2 benchmark interpretation is not contaminated by Layer 3 or Layer 4 overhead unless explicitly labeled.

7.3 Preserve Layer 3 benchmark separation
	•	Confirm batching/speculative/prompt-cache orchestration benchmarks remain separate from Layer 1 and Layer 2 benchmarks.
	•	Confirm orchestration cost is attributable to Layer 3.

7.4 Preserve Layer 4 benchmark separation
	•	Confirm server/streaming/API compatibility/observability overhead is benchmarked separately from Layers 1–3.
	•	Confirm product-surface benchmarks do not redefine runtime-layer success criteria.

7.5 Verify cost attribution after full integration
	•	Confirm Layer 1 cost is still interpretable as runtime-core cost.
	•	Confirm Layer 2 cost is still interpretable as semantic-enrichment cost.
	•	Confirm Layer 3 cost is still interpretable as orchestration cost.
	•	Confirm Layer 4 cost is still interpretable as product-surface cost.
	•	Confirm no hidden cost migration occurred during integration.

⸻

8. Compatibility-reduction checklist

8.1 Decide which compatibility shims are still necessary

For each remaining compatibility shim, confirm:
	•	which exact product surface still depends on it,
	•	which lower-layer boundary it protects,
	•	whether the canonical path still traverses it,
	•	whether removal would now be safe.

8.2 Decide which must be removed immediately

Remove immediately any compatibility shim that:
	•	no longer protects a real dependency,
	•	keeps the old universal runtime boundary alive,
	•	widens lower-layer contracts,
	•	contaminates benchmark interpretation,
	•	obscures canonical path clarity.

8.3 Prevent temporary structures from becoming permanent
	•	Confirm every surviving compatibility structure is explicitly marked as temporary or as a permanent Layer 4 compatibility surface.
	•	Confirm no transitional structure remains unlabeled.
	•	Confirm no transitional structure is treated as canonical by default.

8.4 Final compatibility posture
	•	Confirm that compatibility surfaces remaining after Phase 6 are truly Layer 4-owned and no longer transitional.
	•	Confirm no compatibility mechanism governs internal architecture.

⸻

9. Operational hardening checklist

9.1 Coherence of integrated paths
	•	Confirm canonical paths through Layers 1–4 are coherent end-to-end.
	•	Confirm non-canonical or compatibility paths are explicitly identifiable.
	•	Confirm there is no ambiguity about which path a given surface is using.

9.2 Stability of exposed surfaces
	•	Confirm Layer 4 exposed surfaces are stable enough for ordinary maintenance rather than continued architectural churn.
	•	Confirm lower-layer contracts are stable enough that further work is feature or optimization work, not architectural rescue.

9.3 Clarity of canonical vs non-canonical paths
	•	Confirm canonical runtime path is obvious.
	•	Confirm canonical semantic path is obvious.
	•	Confirm canonical orchestration path is obvious.
	•	Confirm canonical product-surface path is obvious.
	•	Confirm legacy or fallback paths are clearly marked non-canonical.

9.4 Reduction of operational ambiguity
	•	Confirm there is no operational ambiguity about which code area owns a concern.
	•	Confirm maintainers can answer “which layer owns this?” for all major responsibilities.
	•	Confirm old architecture questions no longer need to be re-litigated to work on the system.

⸻

10. Final validation checklist

10.1 Final architectural validation
	•	Validate that all four layers exist and are distinct.
	•	Validate that the actual code matches the canonical layer responsibilities.
	•	Validate that Layer 4 is the only composition root.
	•	Validate that no hidden legacy architecture remains canonical in practice.

10.2 Final benchmark validation
	•	Validate Class A as a pure Layer 1 benchmark under the Benchmark Protocol.
	•	Validate Layer 2 benchmarks as separate and interpretable.
	•	Validate Layer 3 benchmarks as separate and interpretable.
	•	Validate Layer 4/product-surface benchmarks as separate and interpretable.
	•	Validate cost attribution across all benchmark classes.

10.3 Final baseline-strategy validation
	•	Validate that stable non-runtime substrate was actually retained where intended.
	•	Validate that the runtime-core subtree was actually replaced fresh rather than refactored in place under old boundaries.
	•	Validate that retained code did not re-impose old architecture.

10.4 Final validation against the old runtime center
	•	Validate that the old universal generate/decode center is no longer canonical in theory.
	•	Validate that it is no longer canonical in practice.
	•	Validate that no compatibility path still routes most execution through it.
	•	Validate that no maintainer would reasonably treat it as the real architecture anymore.

⸻

11. Phase 6 exit criteria

Phase 6 is complete only when all of the following are true:
	•	Transitional structures have been identified and classified.
	•	Unnecessary shims/adapters/bridges have been reduced or removed.
	•	Legacy contamination residue has been removed or isolated out of canonical paths.
	•	Layers 1–4 are verifiably distinct and canonical.
	•	Layer 4 is verifiably the only composition root.
	•	Lower-layer contracts remain intact and unwidened.
	•	Benchmark separation remains correct after full integration.
	•	Cost attribution remains correct after full integration.
	•	Remaining compatibility surfaces are justified, Layer 4-owned, and no longer architecturally dangerous.
	•	The old universal runtime center is no longer canonical in practice or theory.

Phase 6 is not complete if any of the above remains unresolved.

⸻

12. Refactor completion criteria

The refactor as a whole is complete only when all of the following are true.

12.1 Architecturally complete
	•	The implemented system matches the canonical 4-layer architecture.
	•	Layer responsibilities match the canonical specs.
	•	Layer boundaries are real in code and operation, not only in documentation.
	•	Layer 4 is the sole composition root.
	•	The old universal runtime architecture has been superseded completely.

12.2 Benchmark-valid
	•	Canonical Class A validation exists and is reproducible.
	•	Lower-layer benchmark classes remain separable and interpretable.
	•	Product-surface benchmark classes remain separable and interpretable.
	•	Runtime and product costs are not conflated.

12.3 Compatibility reduction sufficiently closed
	•	Remaining compatibility surfaces are explicitly justified.
	•	Transitional compatibility structures are removed or clearly quarantined.
	•	No temporary architecture bridge remains accidentally permanent.

A refactor that is architecturally complete but not benchmark-valid is not fully complete.
A refactor that is benchmark-valid but still governed by transitional compatibility structures is not fully complete.

⸻

13. Post-refactor handoff

13.1 What remains as ordinary maintenance

After refactor completion, ordinary maintenance may include:
	•	performance tuning within the canonical architecture,
	•	feature work within the canonical layer boundaries,
	•	new benchmark cases or benchmark refinement consistent with the Benchmark Protocol,
	•	product-surface evolution consistent with Layer 4 constraints.

13.2 What remains monitorable but non-blocking

The following may remain monitorable without blocking refactor closure:
	•	performance improvements within a stable benchmark class,
	•	further compatibility reduction that is no longer architecturally risky,
	•	observability refinement,
	•	product-surface ergonomics,
	•	internal cleanup that does not alter canonical layer boundaries.

13.3 What must not be reopened as a basic architectural question

The following must not be reopened as foundational architecture questions unless new evidence is strong enough to justify a new architecture cycle:
	•	whether MLXs should remain performance-core-first,
	•	whether the system should remain 4-layered,
	•	whether Layer 1 should remain sovereign,
	•	whether Layer 2 semantics should remain separate,
	•	whether Layer 3 orchestration should remain separate,
	•	whether Layer 4 should remain the sole composition root,
	•	whether the old universal generate/decode center should return.

These questions are closed by the completed refactor.

⸻

This checklist is now the execution artifact for Phase 6 and the final operational document needed to close the refactor with integration hardening, cleanup, and completion discipline.