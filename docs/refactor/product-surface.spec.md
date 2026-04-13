MLXs — Product Surfaces Spec

Status: canonical subsystem specification
Role: defines Layer 4 as the product-surface and composition-root layer above the Performance Core, General Path, and Advanced Engines
Upstream inputs:
	•	MLXs — Vision / Architectural Foundations
	•	MLXs Research Dossier
	•	MLXs — Performance Core Spec
	•	MLXs — Benchmark Protocol
	•	MLXs — Repo Baseline Audit
	•	MLXs — General Path Spec
	•	MLXs — Advanced Engines Spec

This document defines the Product Surfaces as Layer 4 of the new MLXs architecture. It specifies how product-facing interfaces consume lower layers without widening or deforming their contracts.

⸻

1. Purpose of the Product Surfaces

The Product Surfaces exist to expose MLXs as a usable product and operational system while preserving the integrity of the lower architecture.

Layer 4 owns:
	•	user-facing and system-facing entry surfaces,
	•	transport and compatibility concerns,
	•	configuration exposure,
	•	observability exposure,
	•	lifecycle and management exposure,
	•	endpoint-facing operational semantics.

It exists separately because:
	•	the Performance Core owns execution discipline and active runtime ownership,
	•	the General Path owns rich single-request generation semantics,
	•	the Advanced Engines own orchestration,
	•	none of those layers should be shaped by transport, API compatibility, CLI ergonomics, or operational surface requirements.

Layer 4 is therefore the only layer allowed to be product-shaped. All lower layers must remain architecture-shaped.

⸻

2. Position in the architecture

2.1 Relationship to Layer 1

Layer 4 may consume Layer 1 only through sanctioned lower-layer boundaries. It must not:
	•	directly redefine core contracts,
	•	introduce product-facing metadata into the core,
	•	force Layer 1 to understand server, CLI, or API concerns.

2.2 Relationship to Layer 2

Layer 4 may consume Layer 2 outputs and semantics, but must not:
	•	redefine Layer 2’s semantic contract,
	•	push API/product shaping into Layer 2,
	•	require Layer 2 to emit product-ready objects by default.

2.3 Relationship to Layer 3

Layer 4 may expose capabilities backed by Layer 3, including:
	•	batching-backed capabilities,
	•	speculative-backed capabilities,
	•	prompt-cache-backed capabilities,
	•	engine-backed admission behavior.

It must not:
	•	absorb Layer 3 orchestration internals,
	•	force Layer 3 to adopt transport semantics,
	•	redefine engine contracts as product contracts.

2.4 Why Layer 4 is the composition root

Layer 4 is the composition root because it is the only layer that may:
	•	wire together lower layers for concrete product behavior,
	•	bind configuration to concrete runtime choices,
	•	expose lower-layer capabilities through transport surfaces,
	•	connect observability and lifecycle management to live runtime instances.

No lower layer is permitted to act as a composition root.

⸻

3. Responsibilities

Layer 4 owns the following responsibilities.

3.1 HTTP / SSE server surfaces

It owns:
	•	HTTP request handling,
	•	endpoint routing,
	•	SSE or equivalent streaming surfaces,
	•	request/response envelope shaping,
	•	transport-specific streaming/event emission.

3.2 CLI surfaces

It owns:
	•	interactive developer-facing command-line usage,
	•	CLI-specific output shaping,
	•	CLI-specific lifecycle of a session,
	•	CLI ergonomics and command semantics.

3.3 API compatibility surfaces

It owns:
	•	external API compatibility layers,
	•	request/response schema compatibility,
	•	versioned surface behavior,
	•	compatibility adapters at the product boundary.

3.4 Configuration surfaces

It owns:
	•	file/env/CLI/user-facing configuration exposure,
	•	configuration override surfaces,
	•	startup-time validation surfaces,
	•	mapping product-visible configuration into lower-layer configuration.

3.5 Observability / metrics / tracing surfaces

It owns:
	•	product-facing metrics exposure,
	•	tracing exposure,
	•	logging/telemetry surfaces,
	•	export and integration surfaces for operational observability.

For cache-backed serving features, Layer 4 observability may expose cheap,
operator-meaningful cache state snapshots such as hit/miss counts, entry count,
and approximate resident bytes, provided those remain observational and do not
move cache policy or ownership into Layer 4.

For live serving, Layer 4 observability may also expose cheap operator-visible
request-flow state such as active/pending request counts and outcome counters
(completed, rejected, timed out), provided those remain observational and do
not redefine Layer 3 scheduling or Layer 4 admission policy.

3.6 Lifecycle and model/session management surfaces

It owns:
	•	model loading exposure,
	•	model unload/reload surface behavior,
	•	session/request lifecycle surfaces,
	•	product-facing management commands or endpoints,
	•	exposure of management operations.

Layer 4 readiness/lifecycle exposure must reflect materially usable runtime
state. A product surface must not report itself ready if model residency still
depends on a first-request lazy load.

3.7 Endpoint-facing admission / rejection / timeout semantics

Where these are product-surface semantics, Layer 4 owns:
	•	endpoint-facing queue/reject responses,
	•	request timeout exposure,
	•	endpoint-facing overload behavior,
	•	translation of lower-layer admission/backpressure into user-visible behavior.

⸻

4. Non-responsibilities

Layer 4 must not own the following.

4.1 Core execution discipline

It must not own:
	•	eval/materialization placement inside the core,
	•	core stream discipline,
	•	core token-selection contract,
	•	active cache mutation semantics,
	•	core performance-path control flow.

4.2 Rich single-request generation semantics

It must not own:
	•	stop semantics as a Layer 2 concern,
	•	finish-reason mapping as a Layer 2 concern,
	•	penalties/processors as Layer 2 semantics,
	•	logprob/top-logprob shaping as a Layer 2 concern.

It may expose them, but it does not define them.

4.3 Advanced orchestration internals

It must not own:
	•	batching internals,
	•	speculative acceptance/rejection logic,
	•	prompt-cache orchestration policy internals,
	•	engine scheduling internals,
	•	engine coordination state.

It may call them and expose their capabilities.

4.4 Lower-layer state ownership

It must not own:
	•	core-owned active generation state,
	•	core-owned active cache state,
	•	Layer 2 semantic state,
	•	Layer 3 orchestration-internal ownership.

Layer 4 may observe, request, or expose operations on these through lower-layer contracts, but it must not absorb ownership.

⸻

5. Boundary with the Performance Core

5.1 What Layer 4 may not ask of the core

Layer 4 may not ask Layer 1 to:
	•	emit transport-shaped events,
	•	emit API-compatible envelopes,
	•	own endpoint-facing timeout or retry semantics,
	•	own logging/metrics payloads,
	•	own CLI-facing rendering structures,
	•	widen its output contract for compatibility reasons.

5.2 Preventing server/API concerns from widening Layer 1

The following rules are binding:
	1.	Product-surface compatibility must be implemented at Layer 4, not by extending the core.
	2.	Server-specific requirements must not appear in Layer 1 contracts.
	3.	Product-surface metadata must not become mandatory Layer 1 metadata.
	4.	Lower-layer execution cost must remain independent of Layer 4’s chosen exposure surfaces.

5.3 Core-facing interaction rule

Layer 4 may reach Layer 1 only through lower-layer composition paths that preserve the core contract. It may not bypass Layer 2/3 abstractions in ways that redefine Layer 1.

⸻

6. Boundary with the General Path

6.1 How Layer 4 consumes semantic outputs

Layer 4 may consume:
	•	rich generation-facing outputs from Layer 2,
	•	finish reasons,
	•	optional score/logprob structures,
	•	richer token/step semantics.

6.2 What Layer 4 may not do to Layer 2

Layer 4 may not:
	•	redefine Layer 2 semantics to match product API quirks,
	•	require Layer 2 to expose transport-ready response objects,
	•	push compatibility-layer fields into Layer 2 contracts,
	•	conflate semantic shaping with API shaping.

6.3 Preventing API/product shaping from re-entering Layer 2

The following rules are binding:
	1.	Generation semantics are defined by Layer 2, not by product compatibility requirements.
	2.	Product-facing schema mapping belongs in Layer 4.
	3.	Layer 2 outputs may be adapted by Layer 4, but not reauthored by it.
	4.	If a product surface needs a different representation, the adaptation is Layer 4’s responsibility.

⸻

7. Boundary with Advanced Engines

7.1 Exposing engine-backed capabilities

Layer 4 may expose:
	•	batching-backed endpoints or commands,
	•	speculative-backed endpoints or commands,
	•	prompt-cache-backed endpoints or commands,
	•	engine-backed admission/backpressure behavior.

7.2 What must remain in Layer 3

The following remain in Layer 3:
	•	batching internals,
	•	scheduler internals,
	•	speculative coordination internals,
	•	prompt-cache orchestration policy internals,
	•	engine-local admission internals.

7.3 Boundary rule

Layer 4 may surface engine capabilities, but it must not force Layer 3 to adopt product transport or compatibility semantics.

⸻

8. Surface contract model

8.1 Request / input surface contracts

Layer 4 owns product-facing input contracts, including:
	•	HTTP request schemas,
	•	CLI command/input schemas,
	•	compatibility-layer request objects,
	•	configuration-surface input forms.

These contracts must map into lower-layer contracts through adapters or composition, without widening lower-layer contracts.

8.2 Response / output surface contracts

Layer 4 owns:
	•	API response schemas,
	•	CLI output forms,
	•	transport-facing response payloads,
	•	compatibility-layer output envelopes.

These are Layer 4 contracts, not runtime contracts.

8.3 Streaming / event surface contracts

Layer 4 owns:
	•	streaming payload shapes,
	•	SSE/event framing,
	•	event emission contracts visible to external consumers,
	•	compatibility-layer stream event forms.

The lower layers may provide semantic outputs, but streaming/event surface contracts belong here.

8.4 Config surface contracts

Layer 4 owns:
	•	user-facing config schema exposure,
	•	env/file/CLI override surfaces,
	•	startup-time validation surfaces for product configuration,
	•	mapping into lower-layer configuration shapes.

⸻

9. Server surface model

9.1 Responsibility of the server layer

The server surface is the product-facing HTTP/streaming surface of MLXs.

It owns:
	•	endpoint definitions,
	•	request validation at the product boundary,
	•	response/stream shaping,
	•	endpoint-visible timeout/reject behavior,
	•	transport-specific lifecycle.

9.2 Meaning of “server” in this architecture

The server is a Layer 4 consumer and composition root.
It is not:
	•	the runtime,
	•	the generation semantics layer,
	•	the orchestration layer.

It hosts those capabilities; it does not define them.

9.3 Separation from runtime architecture

The server must remain separable from the runtime architecture by these rules:
	•	runtime capability is exposed upward, not defined downward;
	•	endpoint concerns stay outside Layers 1–3;
	•	transport framing never becomes part of lower-layer contracts.

⸻

10. CLI surface model

10.1 Role of the CLI

The CLI is a product/developer surface for interacting with MLXs outside server transport.

It owns:
	•	command semantics,
	•	interactive input/output behavior,
	•	CLI rendering and interaction modes,
	•	CLI lifecycle around a product-visible session.

10.2 Consumption of lower layers

The CLI may consume:
	•	Layer 2 for rich single-request generation,
	•	Layer 3 for advanced engine-backed flows where appropriate,
	•	Layer 1 only via sanctioned lower-layer composition.

It must not redefine the lower layers for convenience.

⸻

11. Configuration model

11.1 Where configuration lives

Configuration exposure lives in Layer 4.

Layer 4 owns:
	•	product-facing config schema exposure,
	•	layered override surfaces,
	•	validation surfaces,
	•	runtime binding of config to concrete lower-layer instances.

11.2 Preventing configuration from becoming a universal context

The following rules are binding:
	1.	Configuration must not become a universal runtime context object crossing all layers.
	2.	Lower layers receive only the configuration fragments relevant to their contracts.
	3.	Product-facing configuration breadth must be reduced before it reaches lower layers.
	4.	No lower layer may be forced to understand full product config shape.

11.3 Override surfaces

Layer 4 owns the user-facing override mechanisms and is responsible for translating them into lower-layer configuration without leaking Layer 4 shape into lower layers.

⸻

12. Observability model

12.1 Metrics

Layer 4 owns:
	•	metrics exposure,
	•	metrics export surfaces,
	•	operational metric endpoints or interfaces.

It does not own the definition of lower-layer execution behavior, only its exposure and product-surface integration.

12.2 Tracing

Layer 4 owns:
	•	tracing exposure,
	•	trace export/integration surfaces,
	•	product-facing correlation or trace context policies.

It must not require lower layers to emit product-specific tracing structures.

12.3 Logging / telemetry surfaces

Layer 4 owns:
	•	operational logging exposure,
	•	telemetry export surfaces,
	•	user/system-facing diagnostic surfaces.

12.4 Observing without contaminating benchmarks or core

The following rules are binding:
	•	observability surfaces must be disableable or separable from benchmark-critical runs;
	•	benchmark and runtime costs attributable to observability must remain visible as Layer 4 costs, not be misattributed to lower layers;
	•	diagnostic export mechanisms that materially alter execution must not be treated as free.

This aligns with the Benchmark Protocol requirement that observability and diagnostics not pollute the canonical fast-path benchmark.

⸻

13. Lifecycle and management model

13.1 Model loading exposure

Layer 4 owns product-facing exposure of:
	•	model load/unload/reload operations,
	•	management commands or endpoints related to model availability,
	•	lifecycle controls visible to operators or users.

It does not redefine the lower-layer loading/runtime contracts.

13.2 Session / request lifecycle surfaces

Layer 4 owns:
	•	request/session lifecycle as visible externally,
	•	endpoint/session lifecycle semantics,
	•	user/operator-visible session management surfaces.

It does not own lower-layer runtime state ownership.

13.3 Endpoint-facing admission / timeout / rejection semantics

Layer 4 owns:
	•	endpoint-visible timeout behavior,
	•	endpoint-visible rejection behavior,
	•	endpoint-visible overload signaling,
	•	endpoint-visible queue/admission semantics.

13.4 Distinction between product policy and engine/runtime policy

The distinction is binding:
	•	engine/runtime admission and coordination policy stays in Layer 3,
	•	endpoint-visible exposure of those policies stays in Layer 4.

Layer 4 may translate Layer 3 decisions into product-facing behavior, but does not absorb Layer 3 internals.

⸻

14. Compatibility surface policy

14.1 Treatment of API compatibility surfaces

Compatibility surfaces are permitted and belong to Layer 4.

They may provide:
	•	schema compatibility,
	•	request/response compatibility,
	•	transport/event compatibility,
	•	versioned external behavior.

14.2 Preventing compatibility from governing internal architecture

The following rules are binding:
	1.	Compatibility is implemented as a surface adaptation, not an internal architecture driver.
	2.	Lower-layer contracts must not be widened to mirror external API quirks.
	3.	Compatibility layers may map into the internal architecture, but may not redefine it.
	4.	If compatibility requires extra shaping, the cost belongs to Layer 4.

⸻

15. Cost discipline

15.1 Preventing Layer 4 costs from contaminating lower-layer benchmarks

Layer 4 costs must remain separate from runtime costs.

That includes:
	•	transport framing,
	•	compatibility translation,
	•	config processing,
	•	observability export,
	•	endpoint lifecycle handling.

15.2 Separation of product costs and runtime costs

The following rules are binding:
	1.	Product-surface costs must not be reported as if they were Performance Core costs.
	2.	Product-surface enrichments must not widen Layer 1 or Layer 2 contracts by default.
	3.	Product-facing optionality must not become lower-layer mandatory work.
	4.	Benchmark classes must continue to isolate Layer 4 from the canonical fast-path benchmark.

15.3 Visibility of Layer 4 overhead

Layer 4 overhead should remain attributable and measurable as Layer 4 overhead, not hidden inside the runtime layers.

⸻

16. Abstraction policy

16.1 Allowed abstractions

Layer 4 may use abstractions that improve:
	•	composition,
	•	product-surface clarity,
	•	compatibility-layer management,
	•	transport decoupling,
	•	observability exposure,
	•	lifecycle management clarity.

Examples:
	•	request/response surface models,
	•	compatibility adapters,
	•	config mappers,
	•	transport abstractions,
	•	observability exporters.

16.2 Disallowed abstractions

Disallowed:
	•	abstractions that force lower layers to adopt Layer 4 semantics,
	•	product-wide “universal context” objects injected through all layers,
	•	compatibility-first abstractions that govern lower-layer contracts,
	•	abstractions that obscure where Layer 4 overhead begins.

16.3 Acceptance rule

A Layer 4 abstraction is acceptable only if:
	1.	it isolates a true product-surface concern,
	2.	it does not widen Layers 1–3,
	3.	it does not hide Layer 4 overhead as lower-layer overhead,
	4.	it keeps Layer 4 as composition root rather than runtime owner.

⸻

17. Architectural invariants

The following invariants are non-negotiable for Product Surfaces:
	1.	Product Surfaces are Layer 4 and the only composition root.
	2.	Layer 4 may consume Layers 1–3 but may not redefine them.
	3.	Transport, API compatibility, CLI, config exposure, observability exposure, and lifecycle exposure belong to Layer 4.
	4.	Layer 4 must not own core execution discipline.
	5.	Layer 4 must not own rich single-request generation semantics.
	6.	Layer 4 must not own advanced orchestration internals.
	7.	Layer 4 must not own lower-layer runtime state.
	8.	Compatibility surfaces must not govern internal architecture.
	9.	Product-surface costs must not become hidden lower-layer costs.
	10.	No Layer 4 requirement may be allowed to product-shape the Performance Core.

⸻

18. Open questions deferred to later documents

18.1 Deferred to the Implementation Plan

The following are explicitly deferred:
	•	exact module/file decomposition for Layer 4,
	•	migration mechanics from current server/CLI/config surfaces,
	•	compatibility shims,
	•	rollout and cutover sequencing.

18.2 Requiring further audit or operational benchmarking

The following may require additional evidence before final implementation details are frozen:
	•	exact product-surface cost attribution mechanisms,
	•	operational benchmarking for server/streaming overhead,
	•	exact observability-disablement strategy for benchmark purity,
	•	exact endpoint-level timeout/reject semantics relative to Layer 3 behavior.

18.3 Not deferred downward

The following are not delegated to lower layers:
	•	transport semantics,
	•	API compatibility semantics,
	•	product configuration exposure,
	•	observability exposure.

These remain Layer 4 responsibilities.
