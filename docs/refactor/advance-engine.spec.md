MLXs — Advanced Engines Spec

Status: canonical subsystem specification
Role: defines Layer 3 as the advanced orchestration layer above the Performance Core and the General Path
Upstream inputs:
	•	MLXs — Vision / Architectural Foundations
	•	MLXs Research Dossier
	•	MLXs — Performance Core Spec
	•	MLXs — Benchmark Protocol
	•	MLXs — Repo Baseline Audit
	•	MLXs — General Path Spec

This document defines the Advanced Engines as the orchestration layer of the new MLXs architecture. It does not redefine the Performance Core, the General Path, or Product Surfaces.

⸻

1. Purpose of the Advanced Engines

The Advanced Engines exist to own multi-step, multi-request, or coordination-heavy inference orchestration that must remain above the Performance Core and above the General Path.

Their purpose is to provide:
	•	higher-order runtime coordination,
	•	reuse and scheduling policies,
	•	orchestration of advanced inference modes,
	•	controlled composition of multiple generation flows,

without turning the core or the General Path into a universal engine.

The Advanced Engines exist separately because:
	•	the Performance Core owns execution discipline and active generation state,
	•	the General Path owns rich single-request generation semantics,
	•	neither should be widened to absorb cross-request coordination, reuse policy, batching, or speculative orchestration.

Layer 3 therefore exists to centralize orchestration that is real and necessary, while preventing that orchestration from reshaping the contracts of Layers 1 and 2.

⸻

2. Position in the architecture

2.1 Relationship to Layer 1

The Advanced Engines sit above the Performance Core.

They may:
	•	invoke the core repeatedly,
	•	coordinate multiple core sessions,
	•	orchestrate the ordering and grouping of core invocations,
	•	request specific core-compatible execution modes.

They may not:
	•	redefine the core execution model,
	•	widen the core input/output contracts,
	•	impose broad orchestration state inside the core,
	•	move Layer 3 policy into Layer 1.

2.2 Relationship to Layer 2

The Advanced Engines sit above the General Path.

They may:
	•	consume richer single-request generation semantics,
	•	use General Path outputs where orchestration requires richer semantics,
	•	coordinate multiple rich-generation flows.

They may not:
	•	absorb Layer 2 semantics into orchestration primitives,
	•	redefine single-request semantics,
	•	turn orchestration policy into the canonical meaning of generation outputs.

2.3 Relationship to Product Surfaces

The Advanced Engines sit below Product Surfaces.

Product Surfaces may:
	•	call or host the engines,
	•	expose engine-backed capabilities,
	•	apply transport, API, and operational concerns on top of them.

The Advanced Engines may not:
	•	own server transport,
	•	define HTTP/SSE or CLI protocol behavior,
	•	define product-facing request/response envelopes,
	•	absorb API-surface shaping.

⸻

3. Responsibilities

The Advanced Engines own the following responsibilities.

3.1 Batching

They own:
	•	batching policy,
	•	batch formation,
	•	batch membership changes,
	•	scheduling of batched work across compatible generation sessions.

3.2 Continuous batching

They own:
	•	insertion/removal of sessions in active batched execution,
	•	lifecycle management of active batched groups,
	•	reuse of execution windows where continuous batching is supported.

3.3 Speculative decoding orchestration

They own:
	•	draft/target coordination,
	•	speculative acceptance/rejection orchestration,
	•	draft-step planning,
	•	coordination of speculative state across involved runtime components.

3.4 Prompt-cache orchestration

They own:
	•	deciding when prompt-cache lookup is attempted,
	•	deciding when prompt-cache reuse is admissible,
	•	deciding when prompt-cache state should be inserted, updated, or bypassed,
	•	coordinating cache reuse across request boundaries.

3.5 Runtime scheduling

They own:
	•	scheduling policy above single-request generation,
	•	prioritization or selection logic where multiple sessions compete for execution,
	•	orchestration of runtime progression across multiple active inference units.

3.6 Admission / backpressure concerns

Where these concerns are inference-runtime concerns rather than transport concerns, the Advanced Engines may own:
	•	admission decisions for engine-managed workloads,
	•	capacity-aware acceptance/rejection within engine-controlled execution resources,
	•	internal backpressure semantics for orchestration-managed runtime units.

They do not own product-surface or transport-level admission semantics.

3.7 Multi-request coordination

They own:
	•	coordination across multiple generation sessions,
	•	reuse and co-scheduling decisions,
	•	orchestration of shared runtime opportunities,
	•	coordination of multi-session runtime behavior without redefining single-session semantics.

⸻

4. Non-responsibilities

The Advanced Engines must not own the following.

4.1 Core execution discipline

They must not own:
	•	the fundamental decode-step execution model,
	•	eval/materialization policy inside the core,
	•	core cache mutation rules,
	•	core stream discipline,
	•	minimal token-selection contract.

4.2 Single-request rich generation semantics

They must not own:
	•	stop-sequence semantics,
	•	finish-reason shaping for single-request generation,
	•	logprob/top-logprob shaping,
	•	penalties/processors as Layer 2 semantics,
	•	general single-request output shaping.

4.3 Transport / server semantics

They must not own:
	•	HTTP protocol,
	•	SSE/chunk framing,
	•	API response envelopes,
	•	CLI interaction semantics,
	•	transport-layer retry/rejection semantics.

4.4 Product-surface shaping

They must not define:
	•	product-facing event formats,
	•	user-facing stream objects,
	•	observability schemas for product surfaces,
	•	endpoint-level semantics.

4.5 Universal generation contract

They must not become the new “one engine for everything.”
Layer 3 coordinates lower layers; it does not become a replacement for them.

⸻

5. Boundary with the Performance Core

5.1 What engines may ask of the core

The engines may ask the core only for what the core contract already permits:
	•	execution of generation sessions,
	•	progression of active runtime state,
	•	access to explicit core outputs,
	•	access to explicit core-owned state only through sanctioned boundaries.

They may request repeated or coordinated core use.
They may not request a different kind of core.

5.2 What engines may not impose on the core

They may not impose:
	•	batch-specific fields into the core contract,
	•	speculative-specific policy state into the core contract,
	•	prompt-cache policy state into the core contract,
	•	orchestration metadata that is not required by Layer 1,
	•	product-facing lifecycle obligations.

5.3 Preventing orchestration from widening the core

The following rules are binding:
	1.	No engine concern may be pushed into the core merely to avoid orchestration complexity.
	2.	If an orchestration feature needs additional state, that state belongs to the engine unless the Performance Core Spec already grants ownership to Layer 1.
	3.	The core must remain realizable for the single-request fast path without Layer 3 overhead.
	4.	Engine costs must not become implicit default costs of core execution.

⸻

6. Boundary with the General Path

6.1 What engines may consume from the General Path

The engines may consume:
	•	richer single-request generation outputs,
	•	richer finish reasons,
	•	Layer 2 semantic decisions that matter to coordination,
	•	optional semantic shaping relevant to scheduling or orchestration.

6.2 What must remain in Layer 2

The following must remain in Layer 2:
	•	rich single-request stop semantics,
	•	rich single-request logprob/top-logprob shaping,
	•	penalties/processors as generation semantics,
	•	richer sampling semantics,
	•	conversion of core outputs into generation-facing outputs.

6.3 Rules preventing fusion of semantics and orchestration
	1.	If a concern enriches one generation session semantically, it belongs in Layer 2.
	2.	If a concern coordinates multiple sessions or runtime opportunities, it belongs in Layer 3.
	3.	Layer 3 may consume Layer 2 outputs, but must not redefine Layer 2’s semantics contract.
	4.	Layer 2 and Layer 3 must remain independently comprehensible.

⸻

7. Engine contract model

7.1 Input contract

The Advanced Engines input contract may include:
	•	one or more generation-session requests,
	•	references to core/general-path capable execution units,
	•	orchestration policies,
	•	scheduling/admission policies relevant to Layer 3,
	•	prompt-cache orchestration policy,
	•	speculative orchestration policy.

It must not require:
	•	transport envelopes,
	•	server endpoint objects,
	•	product-specific response objects.

7.2 Output contract

The Advanced Engines output contract may include:
	•	orchestrated generation results,
	•	grouped or coordinated session results,
	•	engine-managed progression results,
	•	engine-level lifecycle and coordination outcomes.

It must not hardcode:
	•	HTTP/SSE framing,
	•	API response documents,
	•	product-layer streaming objects.

7.3 Coordination contract

Layer 3 owns the contract describing:
	•	how multiple sessions are coordinated,
	•	how orchestration state progresses,
	•	how engine-level decisions affect session progression,
	•	how core/general-path invocations are scheduled or composed.

7.4 Session / request ownership assumptions

The engines may own:
	•	session coordination state,
	•	orchestration-specific mutable state,
	•	cross-request reuse policy state,
	•	scheduler state,
	•	engine-local admission state.

They do not own:
	•	Layer 1 active cache mutation semantics,
	•	Layer 2 rich single-request semantic definitions,
	•	transport-level request lifecycle semantics.

⸻

8. Batching model

8.1 Responsibility of the batching layer

The batching engine owns:
	•	deciding whether requests can be co-executed,
	•	formation and evolution of execution batches,
	•	lifecycle of batched sessions,
	•	continuous batch maintenance where supported.

8.2 What batching means in this architecture

In this architecture, batching means:
	•	Layer 3 coordination of multiple generation sessions using lower-layer execution capabilities,
	•	not a redefinition of the Performance Core contract,
	•	not a widening of the General Path contract into a “batched universal generation API.”

8.3 Relation to the core

Batching may:
	•	schedule multiple core-compatible units,
	•	map grouped execution opportunities to lower-layer contracts,
	•	manage group membership and progression.

Batching may not:
	•	require the core to understand Layer 3 batch policy,
	•	introduce product semantics into the core,
	•	force every core call site to pay batch-related overhead.

8.4 Continuous batching

Continuous batching remains a Layer 3 concern because it is:
	•	inherently multi-session,
	•	inherently scheduling-driven,
	•	not part of single-request generation semantics,
	•	not part of the minimal runtime core.

⸻

9. Speculative decoding model

9.1 Role of speculative orchestration

Speculative decoding in the new architecture is a Layer 3 orchestration concern.

It owns:
	•	coordination between draft and target execution roles,
	•	acceptance/rejection flow,
	•	advancement and rollback orchestration where needed,
	•	speculative session-level control flow.

9.2 Separation from the core

The core may expose capabilities necessary to support speculation-compatible execution, but:
	•	it must not become speculative-aware by default,
	•	it must not own draft/target policy,
	•	it must not own speculative acceptance semantics.

9.3 Separation from the General Path

The General Path may still define rich single-request semantic outputs used by speculative flows, but:
	•	speculation as a coordination strategy is not Layer 2,
	•	acceptance/rejection orchestration is Layer 3,
	•	multi-model or multi-session coordination is Layer 3.

9.4 What belongs here and what does not

Belongs in Layer 3:
	•	orchestration logic,
	•	speculative progression policy,
	•	draft/target session coordination,
	•	speculative state management above core-local state.

Does not belong in Layer 3:
	•	transport semantics,
	•	product-surface streaming semantics,
	•	core execution discipline,
	•	single-request semantic shaping.

⸻

10. Prompt-cache orchestration model

10.1 Meaning of prompt-cache orchestration

Prompt-cache orchestration means:
	•	deciding if prompt-cache reuse is attempted,
	•	coordinating retrieval and reuse at request/session boundaries,
	•	deciding insertion/update/eviction actions at orchestration level,
	•	mediating reuse policy above the core.

10.2 Ownership rules

The engines may own:
	•	policy state for cache reuse,
	•	reuse decisions,
	•	orchestration of request-to-cache matching,
	•	lifecycle of reusable prompt-cache assets across sessions.

They may not own:
	•	the core’s active cache mutation semantics during an active generation session,
	•	the internal mutation rules of Layer 1 cache state.

10.3 Boundary with cache substrate and core

The cache substrate remains infrastructure.
The Performance Core owns active-request cache use.
Layer 3 owns reuse policy and orchestration across requests.

This distinction is binding:
	•	active runtime ownership stays in Layer 1,
	•	cross-request reuse policy stays in Layer 3.

⸻

11. Scheduling and admission model

11.1 What belongs to Layer 3

Layer 3 may own:
	•	scheduling among engine-managed workloads,
	•	capacity-aware engine admission,
	•	local backpressure within engine-controlled execution resources,
	•	policies for coordinating competing runtime opportunities.

11.2 What does not belong here yet

The following stay outside Layer 3 until Product Surfaces:
	•	endpoint-facing admission semantics,
	•	transport-level queue/reject responses,
	•	user-visible rate limiting semantics,
	•	product-facing timeout and worker model behavior.

11.3 Layer distinction

Layer 3 owns orchestration-internal scheduling and admission.
Layer 4 will own product-surface and transport exposure of those policies.

⸻

12. Cost discipline

12.1 Orchestration without structural core overhead

The Advanced Engines must coordinate lower layers without turning orchestration costs into:
	•	default costs of the single-request path,
	•	hidden costs in Layer 1,
	•	mandatory semantic work in Layer 2.

12.2 No default tax on single-request generation

If Layer 3 is not used, a single-request flow through Layers 1–2 must not pay Layer 3 structural overhead by default.

12.3 Cost visibility

Layer 3 must keep its coordination costs visible:
	•	queueing,
	•	scheduling,
	•	grouping,
	•	reuse checks,
	•	speculative coordination,
	•	cache orchestration.

These costs must not be hidden as if they were intrinsic properties of the core runtime.

12.4 Benchmark relevance

This cost discipline is required by the canonical Benchmark Protocol:
	•	Layer 3 benchmark classes must remain separate from the canonical fast-path benchmark,
	•	orchestration costs must be attributable to Layer 3 rather than interpreted as core cost.

⸻

13. Abstraction policy

13.1 Allowed abstractions

The Advanced Engines may use abstractions that improve:
	•	orchestration clarity,
	•	scheduling composability,
	•	ownership separation,
	•	testability of engine policies,
	•	separation between engine types.

Examples of acceptable abstraction categories:
	•	scheduler policies,
	•	engine-local session coordinators,
	•	orchestration state carriers,
	•	engine capability contracts,
	•	prompt-cache orchestration policies.

13.2 Disallowed abstractions

Disallowed:
	•	abstractions that force Layer 1 or Layer 2 to adopt Layer 3 concepts,
	•	universal runtime containers that merge core, semantics, and engine state,
	•	product-surface abstractions introduced prematurely into Layer 3,
	•	engine contracts so broad that they become a new universal generation interface.

13.3 Acceptance rule

An abstraction in Layer 3 is acceptable only if:
	1.	it isolates real orchestration responsibility,
	2.	it does not widen lower-layer contracts,
	3.	it does not hide engine costs as lower-layer costs,
	4.	it does not absorb Product Surface concerns,
	5.	it keeps Layer 3 separable from Layers 1, 2, and 4.

⸻

14. Architectural invariants

The following invariants are non-negotiable for the Advanced Engines:
	1.	The Advanced Engines are Layer 3 and remain above the Performance Core and General Path.
	2.	Layer 3 owns orchestration, not core execution discipline.
	3.	Layer 3 owns multi-session or multi-step coordination concerns above single-request semantics.
	4.	Layer 3 does not redefine the Performance Core contract.
	5.	Layer 3 does not redefine the General Path contract.
	6.	Batching belongs to Layer 3.
	7.	Speculative orchestration belongs to Layer 3.
	8.	Prompt-cache orchestration belongs to Layer 3.
	9.	Engine-level scheduling and local admission belong to Layer 3 only insofar as they are orchestration concerns, not transport concerns.
	10.	Product-surface shaping must not enter Layer 3.
	11.	Layer 3 must not become a replacement for the rejected universal generation engine.
	12.	If Layer 3 is not used, Layer 1 + Layer 2 must remain viable without paying Layer 3 structural overhead.

⸻

15. Open questions deferred to later specs

15.1 Deferred to Product Surfaces Spec

The following are explicitly deferred:
	•	how server/CLI surfaces expose engine-backed capabilities,
	•	transport-level queue/reject behavior,
	•	endpoint-facing timeout semantics,
	•	product-facing streaming/event schemas,
	•	operational surface around engine scheduling/admission.

15.2 Deferred to the Implementation Plan

The following are explicitly deferred:
	•	exact module/file decomposition for Layer 3,
	•	migration sequence from old batching/speculative/prompt-cache code,
	•	compatibility shims with old interfaces,
	•	rollout ordering across engine types.

15.3 Deferred to further benchmarking and audit

The following may require additional evidence before being frozen in detail:
	•	exact benchmark subclasses for Layer 3 engine families,
	•	exact cost model and success criteria for batching/speculative/prompt-cache reuse,
	•	final division between engine-local admission and product-surface admission,
	•	exact reuse of existing repository batching and prompt-cache code after baseline verification.

⸻