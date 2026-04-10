MLXs — Phase 4 Execution Checklist

Status: execution artifact
Role: operational checklist for Phase 4 of the MLXs refactor

⸻

1. Purpose of Phase 4

Phase 4 exists to reintroduce Layer 3 / Advanced Engines above the already-established Layers 1–2.

It must make four things true:
	•	batching, speculative orchestration, and prompt-cache orchestration exist again,
	•	they exist as a distinct Layer 3 rather than as leakage into Layer 1 or Layer 2,
	•	the sovereignty of Layer 1 and Layer 2 boundaries remains intact,
	•	orchestration does not become a new universal runtime root.

Phase 4 comes after Layer 2 because Layer 3 must consume:
	•	a stable Performance Core,
	•	a stable General Path,
	•	and clear lower-layer boundaries.

If Layer 3 is reintroduced before Layer 2 is stable, orchestration tends to absorb semantics, recreate broad generation contracts, and re-contaminate the runtime center.

⸻

2. Phase 4 goals
	•	Attach Layer 3 above Layers 1–2 without redefining either layer.
	•	Restore advanced orchestration capabilities:
	•	batching,
	•	continuous batching,
	•	speculative decoding orchestration,
	•	prompt-cache orchestration,
	•	engine-level scheduling and admission where appropriate.
	•	Keep Layer 3 clearly separated from:
	•	core execution discipline,
	•	single-request semantic shaping,
	•	product surfaces.
	•	Preserve the canonical Class A benchmark path as a pure Layer 1 benchmark.
	•	Preserve separation between:
	•	core benchmarks,
	•	Layer 2 benchmarks,
	•	Layer 3 benchmarks.

⸻

3. Layer 3 attachment checklist

3.1 Attach Layer 3 above Layers 1–2
	•	Confirm Layer 3 is implemented/reattached as a consumer of Layers 1–2, not as a rewrite of either.
	•	Confirm Layer 3 depends on the frozen Layer 1 and Layer 2 contracts.
	•	Confirm Layer 3 is not introduced by widening Layer 1 or Layer 2 contracts.

3.2 Keep boundaries explicit
	•	Confirm Layer 1 input/output/state contracts remain unchanged after Layer 3 attachment.
	•	Confirm Layer 2 semantic contracts remain unchanged after Layer 3 attachment.
	•	Confirm Layer 3 receives only what Layers 1–2 already permit.
	•	Confirm Layer 3 orchestration state is not forced into lower-layer contracts.

3.3 Prevent Layer 3 from becoming runtime root
	•	Confirm Layer 3 is not the new canonical runtime root.
	•	Confirm Layer 1 remains the sovereign runtime core.
	•	Confirm Layer 2 remains the sovereign rich single-request semantic layer.
	•	Confirm Layer 3 is orchestration only, not a replacement for the rejected universal engine.
	•	Confirm no old mixed runtime/orchestration/generation boundary is being restored under a new name.

⸻

4. Advanced Engines responsibility checklist

Confirm Layer 3 is responsible for the following and only the following categories of concern.

4.1 Batching
	•	Reintroduce batching as a Layer 3 concern.
	•	Confirm batch formation and batch membership are Layer 3-owned.
	•	Confirm batching does not redefine Layer 1 execution discipline.

4.2 Continuous batching
	•	Reintroduce continuous batching as a Layer 3 concern.
	•	Confirm active batch lifecycle and membership evolution are Layer 3-owned.
	•	Confirm continuous batching is not treated as an intrinsic property of Layer 1 or Layer 2.

4.3 Speculative decoding orchestration
	•	Reintroduce speculative decoding orchestration in Layer 3.
	•	Confirm draft/target coordination is Layer 3-owned.
	•	Confirm acceptance/rejection orchestration is Layer 3-owned.
	•	Confirm Layer 3 does not push speculative policy into Layer 1 or Layer 2.

4.4 Prompt-cache orchestration
	•	Reintroduce prompt-cache orchestration in Layer 3.
	•	Confirm reuse/admission/update decisions are Layer 3-owned.
	•	Confirm prompt-cache reuse policy is not encoded into Layer 1.
	•	Confirm prompt-cache orchestration is not silently folded into Layer 2 semantics.

4.5 Scheduling
	•	Reintroduce engine-level scheduling in Layer 3.
	•	Confirm scheduling among engine-managed workloads is Layer 3-owned.
	•	Confirm scheduling is not pushed down into Layer 1 runtime contracts.

4.6 Admission / backpressure as engine concern
	•	Reintroduce engine-local admission/backpressure where appropriate.
	•	Confirm engine-level admission remains distinct from endpoint-facing admission semantics.
	•	Confirm Layer 3 owns only orchestration-internal admission semantics, not Layer 4 transport exposure.

4.7 Multi-request coordination
	•	Reintroduce multi-request coordination in Layer 3.
	•	Confirm all cross-request coordination lives above Layers 1–2.
	•	Confirm Layer 3, not Layer 2, owns reuse and co-scheduling policy across requests.

⸻

5. Advanced Engines non-responsibility checklist

Confirm the following remain outside Layer 3.

5.1 Core execution discipline
	•	Layer 3 does not own Layer 1 eval/materialization/sync policy.
	•	Layer 3 does not own Layer 1 stream discipline.
	•	Layer 3 does not own Layer 1 token-selection contract.
	•	Layer 3 does not own Layer 1 active cache mutation semantics.

5.2 Layer 2 rich single-request semantics
	•	Layer 3 does not own stop semantics as Layer 2 semantics.
	•	Layer 3 does not own finish-reason mapping as Layer 2 semantics.
	•	Layer 3 does not own penalties/processors as Layer 2 semantics.
	•	Layer 3 does not own Layer 2 rich single-request output shaping.

5.3 Server / product transport semantics
	•	Layer 3 does not own HTTP/SSE/CLI/API compatibility semantics.
	•	Layer 3 does not own endpoint request/response shaping.
	•	Layer 3 does not own transport-layer rejection/timeout semantics.

5.4 Product-surface shaping
	•	Layer 3 does not own product-facing events.
	•	Layer 3 does not own API compatibility envelopes.
	•	Layer 3 does not own product observability exports.
	•	Layer 3 does not own configuration exposure surfaces.

⸻

6. Boundary preservation checklist

6.1 Consuming Layers 1–2 without widening them
	•	Confirm Layer 3 uses Layer 1 and Layer 2 contracts as frozen.
	•	Confirm Layer 3 does not add mandatory engine metadata to Layer 1 contracts.
	•	Confirm Layer 3 does not add mandatory engine state to Layer 2 contracts.
	•	Confirm Layer 3 does not require lower layers to understand orchestration-specific state.

6.2 Avoid imposing engine state, metadata, or policy on Layer 1
	•	Confirm batching policy state remains in Layer 3.
	•	Confirm speculative policy state remains in Layer 3.
	•	Confirm prompt-cache orchestration policy state remains in Layer 3.
	•	Confirm scheduler state remains in Layer 3.
	•	Confirm Layer 1 active runtime/cache ownership remains unchanged.

6.3 Avoid fusing semantics and orchestration
	•	Confirm Layer 2 semantic shaping remains distinct from Layer 3 coordination.
	•	Confirm Layer 3 consumes Layer 2 outputs without redefining them.
	•	Confirm no Layer 3 abstraction becomes a new “semantic runtime” boundary.
	•	Confirm no old mixed semantics/orchestration pattern is reintroduced.

⸻

7. Engine ownership checklist

7.1 Coordination-state ownership
	•	Confirm Layer 3 owns coordination state for multi-request or multi-step orchestration.
	•	Confirm this state is not mirrored into Layer 1 or Layer 2.
	•	Confirm coordination state is not hidden in broad lower-layer context objects.

7.2 Ownership of batching/speculative/prompt-cache policy state
	•	Confirm batching policy state is Layer 3-owned.
	•	Confirm speculative orchestration state is Layer 3-owned.
	•	Confirm prompt-cache orchestration state is Layer 3-owned.
	•	Confirm engine-local admission/scheduling state is Layer 3-owned.

7.3 What remains ownership of Layer 1
	•	Active mutable runtime state during generation.
	•	Active mutable cache state during generation.
	•	Layer 1 execution discipline.
	•	Layer 1 minimal token-generation contract.

7.4 What remains ownership of Layer 2
	•	Stop semantics.
	•	EOS/max-token semantic shaping.
	•	Finish-reason mapping.
	•	Logprobs/top-logprobs shaping.
	•	Penalties/processors as single-request semantics.
	•	Richer single-request generation outputs.

⸻

8. Cost-discipline checklist

8.1 No default tax on the single-request path
	•	Confirm Layer 3 introduces no mandatory overhead on pure Layer 1 / Layer 2 single-request flows when Layer 3 is not in use.
	•	Confirm orchestration structures are not initialized or traversed by default in the canonical fast path.
	•	Confirm engine concerns do not become hidden default costs.

8.2 Orchestration visible as Layer 3 cost
	•	Confirm scheduling cost is attributable to Layer 3.
	•	Confirm batching cost is attributable to Layer 3.
	•	Confirm speculative orchestration cost is attributable to Layer 3.
	•	Confirm prompt-cache orchestration cost is attributable to Layer 3.
	•	Confirm engine-local admission/backpressure cost is attributable to Layer 3.

8.3 No contamination of the canonical Class A benchmark
	•	Confirm canonical Class A remains a Layer 1 benchmark.
	•	Confirm Layer 3 is excluded from the canonical Class A path.
	•	Confirm no engine shim or scheduler logic is silently on the Class A path.
	•	Confirm Layer 3 benchmarking is reported separately.

8.4 Separation of engine benchmarks
	•	Confirm batching benchmarks are separate from Class A and Layer 2 benchmarks.
	•	Confirm speculative benchmarks are separate from Class A and Layer 2 benchmarks.
	•	Confirm prompt-cache reuse benchmarks are separate from Class A and Layer 2 benchmarks.
	•	Confirm any engine-level benchmark declares itself as Layer 3.

⸻

9. Allowed reuse checklist for Layer 3

9.1 Components that may be recovered
	•	Recover batching subsystem pieces only if they can be reattached above the new Layers 1–2 contracts.
	•	Recover speculative subsystem pieces only if they can be isolated as orchestration rather than runtime-core behavior.
	•	Recover prompt-cache orchestration pieces only if they remain Layer 3 policy, not Layer 1 ownership.
	•	Recover scheduler/admission helpers only if they remain Layer 3-local.

9.2 Components allowed only with extraction/refactor
	•	Old batching logic coupled to legacy generate boundaries may be reused only after extraction/refactor.
	•	Old speculative logic may be reused only after separating coordination from runtime-core assumptions.
	•	Old prompt-cache integration may be reused only after separating orchestration policy from active cache ownership.
	•	Old runtime-facing engine wrappers may be reused only if stripped of product-surface or semantic-layer assumptions.

9.3 Components that must not re-enter
	•	Old mixed generate/orchestration code paths.
	•	Old universal runtime wrappers that batch/speculate by widening Layer 1 contracts.
	•	Old product-facing scheduler or queue semantics that belong to Layer 4.
	•	Old broad session/context objects that merge Layer 1, Layer 2, and Layer 3 concerns.

⸻

10. Benchmark separation checklist

10.1 Preserve separation between Class A, Layer 2, and Layer 3
	•	Confirm Class A remains a pure Layer 1 benchmark.
	•	Confirm Layer 2 benchmarks remain distinct from Layer 3 benchmarks.
	•	Confirm Layer 3 benchmarks are never reported as if they were core-runtime benchmarks.

10.2 Prevent batching/speculative/prompt-cache from altering Class A interpretation
	•	Confirm batching is not present on the canonical Class A path.
	•	Confirm speculative orchestration is not present on the canonical Class A path.
	•	Confirm prompt-cache reuse orchestration is not present on the canonical Class A path.
	•	Confirm any engine-backed result is labeled as a Layer 3 benchmark result.

10.3 Preserve interpretability of lower-layer benchmarks
	•	Confirm reintroduction of Layer 3 does not force reinterpretation of prior Layer 1 or Layer 2 benchmark results.
	•	Confirm lower-layer benchmark paths remain executable after Layer 3 attachment.
	•	Confirm benchmark reporting still names the layer/class being measured.

⸻

11. Phase 4 validation checklist

11.1 Validate that Layer 3 exists as a real separate layer
	•	Layer 3 exists as a distinct code area or boundary above Layers 1–2.
	•	Layer 3 can be reasoned about without traversing Layer 4 code.
	•	Layer 3 is not merely the old universal runtime with a new label.

11.2 Validate conformance to the Advanced Engines Spec
	•	Layer 3 owns batching.
	•	Layer 3 owns speculative orchestration.
	•	Layer 3 owns prompt-cache orchestration.
	•	Layer 3 owns engine-level scheduling/admission where appropriate.
	•	Layer 3 does not own Layer 1 execution discipline.
	•	Layer 3 does not own Layer 2 rich single-request semantics.
	•	Layer 3 does not own Layer 4 product surfaces.

11.3 Validate sovereignty of Layers 1–2
	•	Layer 1 remains the only sovereign runtime core.
	•	Layer 2 remains the only rich single-request semantic layer.
	•	Layer 3 has not widened Layer 1 contracts.
	•	Layer 3 has not widened Layer 2 contracts.
	•	Layer 3 has not become the de facto runtime root.

11.4 Validate no hidden recontamination
	•	No Layer 4 product transport/config/API/observability logic has crept into Layer 3.
	•	No old mixed runtime/semantics/orchestration boundary has reappeared.
	•	No Layer 3 policy state is hidden in lower-layer abstractions.
	•	No engine concern has become mandatory cost on single-request paths.

⸻

12. Phase 4 exit criteria

Phase 4 is complete only when all of the following are true:
	•	Layer 3 / Advanced Engines exists as a distinct layer above Layers 1–2.
	•	Batching is reintroduced as a Layer 3 concern.
	•	Continuous batching is reintroduced as a Layer 3 concern where applicable.
	•	Speculative decoding orchestration is reintroduced as a Layer 3 concern.
	•	Prompt-cache orchestration is reintroduced as a Layer 3 concern.
	•	Engine-level scheduling/admission is reintroduced only as a Layer 3 concern where appropriate.
	•	Layer 1 remains sovereign and not widened.
	•	Layer 2 remains sovereign and not widened.
	•	Benchmark separation between Layers 1, 2, and 3 is preserved.
	•	No old universal generation engine has been recreated in practice.

Phase 4 is not complete if any of the above remains unresolved.

⸻

13. Immediate handoff to Phase 5

Phase 5 may begin only when the following are true:
	•	Layer 3 exists as a real separate layer above Layers 1–2.
	•	The boundary between orchestration and product surfaces is explicit.
	•	No ambiguity remains about what belongs to Layer 3 vs Layer 4.
	•	Lower-layer benchmark paths remain intact and interpretable.
	•	Layer 3 capabilities are stable enough to be surfaced by Layer 4 without redefining them.
	•	No unresolved question remains about whether transport, API compatibility, CLI, config exposure, observability exposure, or lifecycle exposure belongs in Layer 3. It does not.

Nothing necessary for Product Surfaces implementation may remain ambiguous at handoff.