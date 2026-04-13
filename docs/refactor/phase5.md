MLXs — Phase 5 Execution Checklist

Status: execution artifact
Role: operational checklist for Phase 5 of the MLXs refactor

⸻

1. Purpose of Phase 5

Phase 5 exists to reintroduce Layer 4 / Product Surfaces above the already-established Layers 1–3.

It must make four things true:
	•	server, CLI, configuration, observability, lifecycle, and compatibility surfaces exist again,
	•	they exist as a distinct Layer 4 rather than as contamination of Layers 1–3,
	•	Layers 1–3 remain sovereign in their own boundaries,
	•	Layer 4 becomes the only composition root.

Phase 5 comes after Layer 3 because Product Surfaces must sit above:
	•	a stable Performance Core,
	•	a stable General Path,
	•	a stable Advanced Engines layer.

If Layer 4 is reintroduced too early, product requirements tend to reshape runtime and orchestration contracts, which is explicitly prohibited by the canonical architecture.

⸻

2. Phase 5 goals
	•	Attach Layer 4 above Layers 1–3 without redefining any lower-layer contract.
	•	Restore product-facing surfaces:
	•	HTTP / SSE server,
	•	CLI,
	•	configuration surfaces,
	•	observability surfaces,
	•	lifecycle and management surfaces,
	•	compatibility surfaces.
	•	Ensure Layer 4 becomes the only composition root.
	•	Keep transport/API/config/observability concerns out of Layers 1–3.
	•	Preserve benchmark separation between runtime-layer benchmarks and product-surface benchmarks.
	•	Prepare the system for Phase 6 cleanup and compatibility reduction.

⸻

3. Layer 4 attachment checklist

3.1 Attach Layer 4 above Layers 1–3
	•	Confirm Layer 4 is implemented/reattached as a consumer of Layers 1–3, not as a rewrite of any of them.
	•	Confirm Layer 4 depends on frozen lower-layer contracts rather than widening them.
	•	Confirm product-surface behavior is introduced only at the top of the architecture.

3.2 Keep boundaries explicit
	•	Confirm Layer 1 input/output/state contracts remain unchanged after Layer 4 attachment.
	•	Confirm Layer 2 semantic contracts remain unchanged after Layer 4 attachment.
	•	Confirm Layer 3 orchestration contracts remain unchanged after Layer 4 attachment.
	•	Confirm Layer 4 receives only what Layers 1–3 already permit.

3.3 Prevent Layer 4 from becoming a disguised runtime root
	•	Confirm Layer 4 is the composition root, not the runtime root.
	•	Confirm Layer 1 remains the sovereign runtime core.
	•	Confirm Layer 2 remains the sovereign rich single-request semantic layer.
	•	Confirm Layer 3 remains the sovereign orchestration layer.
	•	Confirm Layer 4 does not recreate a product-shaped universal generation center.

⸻

4. Product Surfaces responsibility checklist

Confirm Layer 4 is responsible for the following and only the following categories of concern.

4.1 HTTP / SSE server surfaces
	•	Reintroduce HTTP server surfaces in Layer 4.
	•	Reintroduce SSE/streaming transport surfaces in Layer 4.
	•	Confirm endpoint routing and transport framing are Layer 4-owned.
	•	Confirm transport-level response shaping is Layer 4-owned.

4.2 CLI surfaces
	•	Reintroduce CLI surfaces in Layer 4.
	•	Confirm command/input/output semantics are Layer 4-owned.
	•	Confirm CLI session/user interaction is Layer 4-owned.

4.3 API compatibility surfaces
	•	Reintroduce API compatibility surfaces in Layer 4.
	•	Confirm request/response compatibility mapping is Layer 4-owned.
	•	Confirm compatibility behavior is implemented as surface adaptation, not lower-layer architecture.

4.4 Configuration surfaces
	•	Reintroduce user-facing configuration surfaces in Layer 4.
	•	Confirm env/file/CLI override exposure is Layer 4-owned.
	•	Confirm startup/config validation surfaces are Layer 4-owned.

4.5 Observability / metrics / tracing surfaces
	•	Reintroduce observability exposure surfaces in Layer 4.
	•	Confirm metrics exposure is Layer 4-owned.
	•	Confirm tracing exposure is Layer 4-owned.
	•	Confirm logging/telemetry export surfaces are Layer 4-owned.
	•	Where useful, confirm low-overhead operator visibility exists for prompt-cache hit/miss/entry/byte state.
	•	Where useful, confirm low-overhead operator visibility exists for live request/queue/outcome state.

4.6 Lifecycle and model/session management surfaces
	•	Reintroduce lifecycle and management exposure in Layer 4.
	•	Confirm model load/unload/reload exposure is Layer 4-owned.
	•	Confirm request/session lifecycle exposure is Layer 4-owned.
	•	Confirm operator/user-facing management operations are Layer 4-owned.
	•	Confirm readiness/health does not report ready before model residency is real.

4.7 Endpoint-facing admission / rejection / timeout semantics
	•	Reintroduce endpoint-facing admission semantics in Layer 4.
	•	Reintroduce endpoint-facing rejection semantics in Layer 4.
	•	Reintroduce endpoint-facing timeout semantics in Layer 4.
	•	Confirm these remain product-surface semantics, not lower-layer runtime policy.

⸻

5. Product Surfaces non-responsibility checklist

Confirm the following remain outside Layer 4.

5.1 Core execution discipline
	•	Layer 4 does not own Layer 1 eval/materialization/sync policy.
	•	Layer 4 does not own Layer 1 stream discipline.
	•	Layer 4 does not own Layer 1 token-selection contract.
	•	Layer 4 does not own active cache mutation semantics.

5.2 Layer 2 rich single-request semantics
	•	Layer 4 does not own stop semantics.
	•	Layer 4 does not own finish-reason mapping.
	•	Layer 4 does not own penalties/processors semantics.
	•	Layer 4 does not own logprobs/top-logprobs semantics.
	•	Layer 4 does not own rich single-request generation semantics.

5.3 Layer 3 orchestration internals
	•	Layer 4 does not own batching internals.
	•	Layer 4 does not own speculative coordination internals.
	•	Layer 4 does not own prompt-cache orchestration internals.
	•	Layer 4 does not own scheduler internals.
	•	Layer 4 does not own engine-local admission/backpressure internals.

5.4 Lower-layer state ownership
	•	Layer 4 does not own Layer 1 active runtime state.
	•	Layer 4 does not own Layer 1 active cache state.
	•	Layer 4 does not own Layer 2 semantic state.
	•	Layer 4 does not own Layer 3 orchestration-internal state.

⸻

6. Boundary preservation checklist

6.1 Consuming Layers 1–3 without widening them
	•	Confirm Layer 4 uses lower-layer contracts as frozen.
	•	Confirm Layer 4 does not add mandatory product metadata to Layer 1 contracts.
	•	Confirm Layer 4 does not add mandatory transport/config/API fields to Layer 2 contracts.
	•	Confirm Layer 4 does not add mandatory product-policy state to Layer 3 contracts.

6.2 Prevent API/product shaping from re-entering Layer 2
	•	Confirm Layer 2 semantic outputs are consumed by Layer 4, not redefined by it.
	•	Confirm API compatibility does not alter Layer 2’s meaning.
	•	Confirm Layer 4 adapts Layer 2 outputs rather than pushing API shape downward.

6.3 Prevent transport/config/observability from re-entering Layers 1–3
	•	Confirm no transport-specific structure enters Layer 1.
	•	Confirm no transport-specific structure enters Layer 2.
	•	Confirm no transport-specific structure enters Layer 3.
	•	Confirm no configuration-universal context object crosses all layers.
	•	Confirm observability export structures remain Layer 4-owned.

⸻

7. Composition-root checklist

7.1 Verify Layer 4 is the only composition root
	•	Confirm Layer 4 is the only layer wiring concrete lower-layer implementations together.
	•	Confirm Layer 1 is not acting as a composition root.
	•	Confirm Layer 2 is not acting as a composition root.
	•	Confirm Layer 3 is not acting as a composition root.
	•	Confirm product-facing bootstrapping and lifecycle composition happen only in Layer 4.

7.2 Prevent hidden composition roots in lower layers
	•	Confirm no lower-layer module is constructing broad product-surface dependencies internally.
	•	Confirm no lower-layer path is doing implicit environment/config resolution as if it were a product entrypoint.
	•	Confirm no lower-layer subsystem has grown into a de facto composition root by convenience.

⸻

8. Configuration and compatibility checklist

8.1 Reintroduce configuration without creating universal contexts
	•	Confirm Layer 4 owns configuration exposure.
	•	Confirm lower layers receive only the configuration fragments relevant to their contracts.
	•	Confirm no full product config object becomes a universal context crossing Layers 1–4.
	•	Confirm override mechanisms are translated at Layer 4 boundaries.

8.2 Reintroduce compatibility surfaces safely
	•	Confirm compatibility surfaces remain Layer 4-owned.
	•	Confirm compatibility adapters map into lower-layer contracts rather than reshaping them.
	•	Confirm external compatibility requirements are not driving internal architecture.

8.3 Prevent external compatibility from governing architecture
	•	Confirm compatibility fields are not pushed into lower-layer canonical contracts.
	•	Confirm lower layers remain architecture-shaped, not API-shaped.
	•	Confirm any compatibility shim is explicitly transitional or explicitly Layer 4-owned.

⸻

9. Observability and cost-discipline checklist

9.1 Keep observability separate from runtime costs
	•	Confirm observability exposure remains Layer 4-owned.
	•	Confirm metrics export cost is attributable to Layer 4.
	•	Confirm tracing export cost is attributable to Layer 4.
	•	Confirm logging/telemetry export cost is attributable to Layer 4.

9.2 Prevent benchmark contamination
	•	Confirm canonical Layer 1 benchmark paths do not traverse Layer 4.
	•	Confirm Layer 2 and Layer 3 benchmarks are not silently polluted by Layer 4 transport/config/observability overhead.
	•	Confirm any Layer 4 product benchmark is reported separately from runtime-layer benchmarks.

9.3 Maintain benchmark purity
	•	Confirm observability can be disabled or separated for benchmark-critical runs.
	•	Confirm product-surface metrics do not alter canonical benchmark interpretation.
	•	Confirm transport and compatibility overhead is never reported as if it were Layer 1/2/3 cost.

⸻

10. Allowed reuse checklist for Layer 4

10.1 Components that may be recovered
	•	Recover server shell components only if they can be reattached above new Layers 1–3 contracts.
	•	Recover CLI shell components only if they remain pure Layer 4 surfaces.
	•	Recover config exposure components only if they do not become universal cross-layer context.
	•	Recover observability surface components only if they stay out of lower-layer contracts.
	•	Recover lifecycle/management surface components only if they remain Layer 4-owned.

10.2 Components allowed only with extraction/refactor
	•	Old server paths may be reused only after removing assumptions about old runtime roots.
	•	Old CLI paths may be reused only after removing direct dependence on legacy generation boundaries.
	•	Old config surfaces may be reused only after proving they can be translated into lower-layer fragments.
	•	Old observability surfaces may be reused only after separating exposure from runtime instrumentation contracts.
	•	Old compatibility surfaces may be reused only after isolating them as Layer 4 adapters.

10.3 Components that must not re-enter
	•	Old product-shaped runtime boundaries.
	•	Old transport-aware runtime helpers that widen Layers 1–3.
	•	Old universal request/session/context objects crossing all layers.
	•	Old compatibility-first abstractions that govern internal layer contracts.
	•	Old server/CLI wiring that bypasses or reshapes lower-layer boundaries.

⸻

11. Benchmark separation checklist

11.1 Preserve separation between runtime-layer and product-surface benchmarks
	•	Confirm Class A remains a pure Layer 1 benchmark.
	•	Confirm Layer 2 benchmarks remain separate from Layer 4 benchmarks.
	•	Confirm Layer 3 benchmarks remain separate from Layer 4 benchmarks.
	•	Confirm Layer 4 benchmarks are explicitly labeled as product-surface benchmarks.

11.2 Prevent server/streaming/API compatibility overhead from altering lower-layer interpretation
	•	Confirm server transport overhead is not included in Layer 1 benchmark interpretation.
	•	Confirm streaming overhead is not included in Layer 1 benchmark interpretation.
	•	Confirm API compatibility overhead is not included in Layer 2/3 benchmark interpretation.
	•	Confirm Layer 4-specific benchmark results do not redefine lower-layer success criteria.

⸻

12. Phase 5 validation checklist

12.1 Validate that Layer 4 exists as a real separate layer
	•	Layer 4 exists as a distinct code area or boundary above Layers 1–3.
	•	Layer 4 can be reasoned about without collapsing into lower-layer code.
	•	Layer 4 is not merely a renamed legacy server/generate center.

12.2 Validate conformance to the Product Surfaces Spec
	•	Layer 4 owns server surfaces.
	•	Layer 4 owns CLI surfaces.
	•	Layer 4 owns compatibility surfaces.
	•	Layer 4 owns configuration surfaces.
	•	Layer 4 owns observability exposure.
	•	Layer 4 owns lifecycle and management exposure.
	•	Layer 4 owns endpoint-facing admission/rejection/timeout semantics.

12.3 Validate sovereignty of Layers 1–3
	•	Layer 1 remains the only sovereign runtime core.
	•	Layer 2 remains the only rich single-request semantic layer.
	•	Layer 3 remains the only advanced orchestration layer.
	•	Layer 4 has not widened any lower-layer contract.
	•	Layer 4 has not become a disguised runtime or orchestration root.

12.4 Validate composition-root correctness
	•	Layer 4 is the only composition root.
	•	No hidden composition root remains in Layers 1–3.
	•	Product wiring is not duplicated below Layer 4.

12.5 Validate no hidden recontamination
	•	No transport semantics have leaked into Layers 1–3.
	•	No compatibility semantics have leaked into Layers 1–3.
	•	No observability-export semantics have leaked into Layers 1–3.
	•	No config-universal context object has re-entered the architecture.

⸻

13. Phase 5 exit criteria

Phase 5 is complete only when all of the following are true:
	•	Layer 4 / Product Surfaces exists as a distinct layer above Layers 1–3.
	•	HTTP/SSE server surfaces are reintroduced as Layer 4 concerns.
	•	CLI surfaces are reintroduced as Layer 4 concerns.
	•	API compatibility surfaces are reintroduced as Layer 4 concerns.
	•	Configuration surfaces are reintroduced as Layer 4 concerns.
	•	Observability/metrics/tracing surfaces are reintroduced as Layer 4 concerns.
	•	Lifecycle and management surfaces are reintroduced as Layer 4 concerns.
	•	Endpoint-facing admission/rejection/timeout semantics are reintroduced as Layer 4 concerns.
	•	Layer 4 is the only composition root.
	•	Layers 1–3 remain sovereign in their boundaries.
	•	Benchmark separation between runtime layers and product surfaces is preserved.
	•	No product-shaped architecture has reappeared underneath Layer 4.

Phase 5 is not complete if any of the above remains unresolved.

⸻

14. Immediate handoff to Phase 6

Phase 6 may begin only when the following are true:
	•	Layers 1–4 all exist as real separate layers.
	•	Temporary shims and compatibility structures still present are explicitly identified.
	•	No ambiguity remains about what is canonical versus transitional.
	•	Lower-layer benchmark paths remain intact and interpretable.
	•	Product surfaces are stable enough for cleanup and compatibility reduction to proceed.
	•	No unresolved question remains about whether Layer 4 owns transport/API/config/observability/lifecycle exposure. It does.

Nothing necessary for integration hardening / cleanup / compatibility reduction may remain ambiguous at handoff.
