MLXs — Phase 3 Execution Checklist

Status: execution artifact
Role: operational checklist for Phase 3 of the MLXs refactor

⸻

1. Purpose of Phase 3

Phase 3 exists to reintroduce Layer 2 / General Path above the validated Layer 1 / Performance Core.

It must make four things true:
	•	rich single-request generation semantics exist again,
	•	they exist as a distinct Layer 2 rather than as contamination inside Layer 1,
	•	the sovereign Layer 1 boundary remains intact,
	•	the canonical Class A benchmark path remains pure and interpretable.

Phase 3 comes after Layer 1 validation because Layer 2 must be attached to a validated core, not co-designed by accident with it. If Layer 2 is reintroduced before the core boundary has been benchmark-aligned, then the fast-path benchmark loses meaning and the old “universal generation engine” failure mode returns.

⸻

2. Phase 3 goals
	•	Attach Layer 2 above Layer 1 without redefining Layer 1.
	•	Restore rich single-request generation semantics.
	•	Reconstruct Layer 2 as a distinct semantic layer, not a hidden extension of the core.
	•	Keep Layer 3 concerns out of Layer 2.
	•	Keep Layer 4 concerns out of Layer 2.
	•	Preserve the canonical Class A benchmark path as a Layer 1 benchmark path.
	•	Ensure optional semantic enrichments remain optional and visible in cost.

⸻

3. Layer 2 attachment checklist

3.1 Attach Layer 2 above Layer 1
	•	Confirm Layer 2 is implemented/reattached as a consumer of Layer 1, not as a rewrite of Layer 1.
	•	Confirm Layer 2 depends on Layer 1 contracts already frozen by the Performance Core Spec.
	•	Confirm Layer 2 is not introduced by modifying Layer 1 to emit Layer 2-shaped outputs by default.

3.2 Keep boundaries explicit
	•	Confirm Layer 1 input boundary remains unchanged by Layer 2 attachment.
	•	Confirm Layer 1 output boundary remains minimal after Layer 2 attachment.
	•	Confirm Layer 1 state/cache ownership remains singular and unchanged.
	•	Confirm Layer 2 receives only what Layer 1 contract already allows.

3.3 Prevent Layer 2 from becoming runtime root
	•	Confirm Layer 2 is not the new canonical runtime root.
	•	Confirm Layer 1 remains the only sovereign runtime core.
	•	Confirm old broad generate()-style shape is not reintroduced as the new Layer 2 boundary.
	•	Confirm Layer 2 is semantically rich but not architecturally central.

⸻

4. General Path responsibility checklist

Confirm Layer 2 is responsible for the following and only the following categories of concerns.

4.1 Stop semantics
	•	Reintroduce rich stop semantics in Layer 2.
	•	Confirm stop-sequence behavior is handled above Layer 1.
	•	Confirm stop semantics do not force broader Layer 1 outputs.

4.2 EOS / max-token policies
	•	Reintroduce richer EOS/max-token policy shaping in Layer 2.
	•	Confirm Layer 2 maps minimal Layer 1 finish state into richer generation-facing semantics.

4.3 Finish-reason mapping
	•	Reintroduce finish-reason mapping in Layer 2.
	•	Confirm Layer 1 remains responsible only for minimal continuation/finish signaling.
	•	Confirm richer finish distinctions are Layer 2-owned.

4.4 Richer single-request generation outputs
	•	Reintroduce richer generation-facing outputs in Layer 2.
	•	Confirm these outputs are not transport/product-shaped.
	•	Confirm they remain Layer 2 semantic outputs, not Layer 4 surface outputs.

4.5 Logprobs / top-logprobs
	•	Reintroduce logprobs/top-logprobs in Layer 2.
	•	Confirm logprob shaping remains optional.
	•	Confirm Layer 1 is not widened to always pay for Layer 2 score shaping.

4.6 Penalties / processors
	•	Reintroduce penalties/processors in Layer 2.
	•	Confirm they are semantic enrichments above Layer 1 rather than Layer 1 invariants.

4.7 Richer sampling behavior
	•	Reintroduce richer single-request sampling behavior in Layer 2.
	•	Confirm this does not change the canonical Layer 1 fast-path contract.
	•	Confirm richer sampling does not become mandatory in Layer 1.

⸻

5. General Path non-responsibility checklist

Confirm the following remain outside Layer 2.

5.1 Batching
	•	Layer 2 does not own batching.
	•	No Layer 3 batching semantics are pulled down into Layer 2.

5.2 Speculative orchestration
	•	Layer 2 does not own speculative coordination.
	•	No draft/target orchestration is introduced as a Layer 2 concern.

5.3 Prompt-cache orchestration
	•	Layer 2 does not own prompt-cache reuse policy.
	•	No cross-request reuse coordination is pulled into Layer 2.

5.4 Server / product transport semantics
	•	Layer 2 does not own HTTP/SSE/CLI/API compatibility shaping.
	•	No Layer 4 transport or compatibility structures are introduced into Layer 2.

5.5 Lower-layer execution discipline
	•	Layer 2 does not own Layer 1 eval/materialization/sync policy.
	•	Layer 2 does not own Layer 1 stream policy.
	•	Layer 2 does not own Layer 1 active cache mutation semantics.

⸻

6. Boundary preservation checklist

6.1 Consuming Layer 1 without widening it
	•	Confirm Layer 2 uses the Layer 1 contract as-is.
	•	Confirm Layer 2 does not add mandatory fields to Layer 1 input/output/state boundaries.
	•	Confirm Layer 2 does not require Layer 1 to carry semantic state that belongs to Layer 2.

6.2 Avoid imposing rich outputs on the core
	•	Confirm Layer 1 is not forced to emit product- or semantics-rich per-step structures by default.
	•	Confirm optional Layer 2 enrichments remain outside the minimal Layer 1 output contract.
	•	Confirm Layer 2 output shaping happens above the Layer 1 boundary.

6.3 Avoid imposing Layer 2 costs on the fast path
	•	Confirm Layer 2 logic is not on the canonical Class A benchmark path unless the benchmark class explicitly includes it.
	•	Confirm richer semantics do not become default costs of the Layer 1 benchmark path.
	•	Confirm Layer 1 remains realizable without Layer 2 enrichments.

⸻

7. Cost-discipline checklist

7.1 Optional enrichments
	•	Confirm logprobs/top-logprobs are optional.
	•	Confirm penalties/processors are optional.
	•	Confirm richer finish/semantic shaping is optional relative to the canonical fast path.
	•	Confirm optional Layer 2 enrichments are explicit in contracts and usage.

7.2 Materialization visibility
	•	Confirm host-side materialization introduced by Layer 2 remains visible.
	•	Confirm no Layer 2 wrapper hides scalar extraction or score materialization costs.
	•	Confirm no silent evaluation boundary is introduced by Layer 2 helpers.

7.3 No hidden host-side work
	•	Confirm Layer 2 does not hide host-side work in “simple” helpers that alter runtime cost materially.
	•	Confirm score shaping / finish shaping / semantic output construction are treated as Layer 2 costs, not Layer 1 costs.

7.4 No contamination of canonical Class A benchmark path
	•	Confirm the canonical Layer 1 benchmark path remains free of Layer 2-only costs unless explicitly running a non-Class-A benchmark.
	•	Confirm any Layer 2 benchmark path is labeled separately from Class A.

⸻

8. Allowed reuse checklist for Layer 2

8.1 Components that may be recovered
	•	Recover old stop semantics only if they can be extracted cleanly into Layer 2.
	•	Recover finish-reason logic only if it can be isolated from product-surface shaping.
	•	Recover logprob-related shaping only if it does not widen Layer 1.
	•	Recover richer sampling/processor logic only if it remains clearly Layer 2.

8.2 Components allowed only with extraction/refactor
	•	Old generation helpers that mixed runtime and semantic logic may be reused only after extraction into true Layer 2 boundaries.
	•	Old generate()-adjacent semantic logic may be reused only if detached from old universal runtime assumptions.
	•	Old output/event shaping helpers may be reused only if stripped of Layer 4 transport semantics.

8.3 Components that must not re-enter
	•	Old broad public generate boundary as canonical Layer 2 root.
	•	Old mixed runtime/semantics helpers that blur Layer 1 and Layer 2.
	•	Old orchestration logic that belongs in Layer 3.
	•	Old product/API shaping that belongs in Layer 4.
	•	Any broad context/recipe/session object that re-aggregates lower and upper layer concerns.

⸻

9. Benchmark separation checklist

9.1 Preserve Class A separation
	•	Confirm canonical Class A remains a Layer 1 benchmark.
	•	Confirm Phase 3 work does not change what Class A means.
	•	Confirm Layer 2 code is not silently on the Class A path.

9.2 Separate Layer 2 benchmark paths
	•	Confirm any benchmark involving logprobs/top-logprobs is labeled as a non-Class-A benchmark.
	•	Confirm any benchmark involving penalties/processors is labeled as a non-Class-A benchmark.
	•	Confirm any benchmark involving richer sampling is labeled as a non-Class-A benchmark.

9.3 Avoid altered interpretation of Layer 1 results
	•	Confirm Layer 2 reintroduction does not force reinterpretation of prior Layer 1 benchmark results.
	•	Confirm Layer 1 benchmark path remains executable after Layer 2 attachment.
	•	Confirm benchmark reports continue to state which layer/class is being measured.

⸻

10. Phase 3 validation checklist

10.1 Validate that Layer 2 exists as a separate layer
	•	Layer 2 exists as a distinct code area or boundary above Layer 1.
	•	Layer 2 can be reasoned about without traversing Layer 3 or Layer 4 code.
	•	Layer 2 is not merely “the top half” of a universal generate engine.

10.2 Validate conformance to the General Path Spec
	•	Layer 2 owns rich single-request generation semantics.
	•	Layer 2 does not own batching/speculative/prompt-cache orchestration.
	•	Layer 2 does not own transport/product shaping.
	•	Layer 2 consumes Layer 1 without redefining it.

10.3 Validate Layer 1 sovereignty
	•	Layer 1 remains the only sovereign runtime core.
	•	Layer 1 contracts remain minimal after Layer 2 attachment.
	•	Layer 1 ownership rules remain unchanged.
	•	Layer 2 has not become the de facto runtime root.

10.4 Validate no hidden recontamination
	•	No Layer 3 orchestration logic has crept into Layer 2.
	•	No Layer 4 transport/config/API logic has crept into Layer 2.
	•	No old universal-generate contract has reappeared under a new name.
	•	No optional Layer 2 enrichment has become mandatory Layer 1 work.

⸻

11. Phase 3 exit criteria

Phase 3 is complete only when all of the following are true:
	•	Layer 2 / General Path exists as a distinct layer above Layer 1.
	•	Rich single-request generation semantics have been reintroduced.
	•	Layer 2 responsibilities match the General Path Spec.
	•	Layer 2 non-responsibilities remain outside Layer 2.
	•	Layer 1 remains sovereign and not redefined.
	•	Canonical Class A benchmark separation is preserved.
	•	Optional Layer 2 enrichments do not impose default cost on the fast path.
	•	No old universal generate/decode boundary has returned as canonical architecture.
	•	Layer 2 is ready to be consumed by Layer 3 without ambiguity.

Phase 3 is not complete if any of the above remains unresolved.

⸻

12. Immediate handoff to Phase 4

Phase 4 may begin only when the following are true:
	•	Layer 2 exists as a real layer above Layer 1.
	•	The boundary between single-request semantics and orchestration is explicit.
	•	No ambiguity remains about what belongs to Layer 2 vs Layer 3.
	•	Canonical Class A benchmark path is still intact and interpretable.
	•	Layer 2 outputs are stable enough to be consumed by Advanced Engines without redefining them.
	•	No unresolved question remains about whether batching/speculative/prompt-cache orchestration belongs in Layer 2. It does not.

Nothing necessary for Advanced Engines implementation may remain ambiguous at handoff.