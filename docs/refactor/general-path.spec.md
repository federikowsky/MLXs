MLXs — General Path Spec

Status: canonical subsystem specification
Role: defines the rich single-request generation layer above the Performance Core
Upstream inputs:
	•	MLXs — Vision / Architectural Foundations
	•	MLXs Research Dossier
	•	MLXs — Performance Core Spec
	•	MLXs — Benchmark Protocol
	•	MLXs — Repo Baseline Audit

This document defines the General Path as the first major layer above the Performance Core. It does not redefine the core contract, batching, speculative orchestration, prompt-cache orchestration, or product-surface transport behavior.

⸻

1. Purpose of the General Path

The General Path exists to provide rich generation semantics above the Performance Core without contaminating the core contract.

Its role in the new MLXs is:
	•	to consume core outputs,
	•	to add generation semantics that are broader than the fast path,
	•	to remain below Advanced Engines and Product Surfaces,
	•	to serve as the canonical single-request rich-generation layer.

It exists separately from the Performance Core because the core is defined around minimal execution discipline, explicit state ownership, and cost visibility under MLX’s lazy execution model, where evaluation and host materialization are real architectural events. The General Path is allowed to add semantic richness, but it must do so as an explicit layer above the core rather than by widening the core itself. MLX explicitly documents that unused outputs still incur graph-build cost and that item(), printing arrays, and host conversion trigger evaluation, which makes this separation technically necessary rather than stylistic.

⸻

2. Position in the architecture

2.1 Relationship to Layer 1

The General Path is Layer 2.

It consumes the Performance Core and may:
	•	enrich core outputs,
	•	impose richer generation semantics,
	•	map minimal core finish signals into richer generation-facing results.

It may not:
	•	redefine core state ownership,
	•	force broader per-step outputs into the core,
	•	make the core aware of product semantics,
	•	make the fast path pay for general-path semantics by default.

2.2 Relationship to Advanced Engines

The General Path sits below Advanced Engines.

Advanced Engines may consume the General Path for richer single-request semantics when needed, but orchestration concerns do not belong here.

The General Path is not:
	•	a batching layer,
	•	a scheduler,
	•	a speculative coordinator,
	•	a prompt-cache orchestrator.

2.3 Relationship to Product Surfaces

The General Path sits below Product Surfaces.

Product Surfaces may consume General Path outputs, but product transport and API semantics are not part of this layer.

The General Path is therefore:
	•	richer than the core,
	•	narrower than engines,
	•	lower than product surfaces.

⸻

3. Responsibilities

The General Path owns the following responsibilities.

3.1 Richer generation semantics

It owns single-request generation semantics that go beyond the minimal fast-path contract.

3.2 Stop semantics

It owns richer stopping behavior, including:
	•	EOS handling beyond the minimal core finish contract,
	•	stop-sequence handling,
	•	max-token policy shaping at the generation layer where the core emits only minimal continuation/finish state.

3.3 EOS / max-token policies

It maps minimal termination state into richer generation-facing termination semantics.

3.4 Logprobs / top-logprobs

It owns:
	•	materialization and shaping of logprobs,
	•	top-logprobs shaping,
	•	optional exposure of richer score outputs for generation consumers.

Because MLX graph outputs incur cost even when not fully used, this functionality must remain optional and layered above the core.

3.5 Penalties / processors

It owns richer generation-time transforms such as:
	•	repetition-related penalties,
	•	other logits processors,
	•	similar semantic transforms that are broader than the minimal token-selection contract.

3.6 Richer sampling behavior

It owns sampling behavior that is broader than the narrowest fast-path selection regime.

This includes:
	•	non-minimal sampling modes,
	•	richer policy combinations,
	•	semantics that require more than the minimal token-selection contract.

3.7 Conversion of core outputs into generation-facing outputs

It owns the mapping from:
	•	minimal core results,
to:
	•	generation-facing results,
	•	rich token or step semantics,
	•	richer finish reasons,
	•	richer metadata.

This conversion must remain above the core and below product transport surfaces.

⸻

4. Non-responsibilities

The General Path must not own the following.

4.1 Batching

It must not own:
	•	prefill batching,
	•	decode batching,
	•	continuous batching,
	•	batch scheduling.

4.2 Speculative orchestration

It must not own:
	•	draft/target model coordination,
	•	acceptance/rejection orchestration,
	•	speculative cache management.

4.3 Prompt-cache orchestration

It must not own:
	•	prompt-cache admission,
	•	prompt-cache eviction,
	•	cache reuse policy across requests,
	•	prompt-cache reuse coordination.

4.4 Server / product transport semantics

It must not own:
	•	HTTP/SSE framing,
	•	OpenAI-compatible response shaping,
	•	product request envelopes,
	•	transport-level streaming protocol,
	•	worker or queue policy.

4.5 Runtime scheduling / admission

It must not own:
	•	queueing,
	•	backpressure,
	•	multi-request scheduling,
	•	runtime admission logic.

4.6 Product observability semantics

It must not define:
	•	server-facing metrics payloads,
	•	product-layer logs or telemetry envelopes,
	•	API-facing trace semantics.

⸻

5. Boundary with the Performance Core

5.1 What the General Path receives from the core

The General Path receives only what the Performance Core contract permits:
	•	core generation results,
	•	minimal finish/continuation state,
	•	minimal metadata exposed by the core contract,
	•	explicit outputs whose cost is already visible and contract-bound.

It does not invent new core outputs.

5.2 What the General Path may enrich

It may enrich:
	•	finish interpretation,
	•	token-level semantics,
	•	stop handling,
	•	score/logprob shaping,
	•	generation metadata,
	•	richer output forms needed by upper layers.

5.3 What the General Path may not impose on the core

It may not impose:
	•	broader mandatory per-step outputs,
	•	product-facing event objects,
	•	orchestration state,
	•	prompt-cache policy state,
	•	batching-related state,
	•	transport-level metadata.

5.4 Rules preventing General Path expansion of the fast path

The following rules are binding:
	1.	The General Path must consume the core; it must not redefine it.
	2.	Any General Path feature that requires additional work must remain optional at the layer boundary.
	3.	The fast path must remain realizable through the core without mandatory General Path overhead.
	4.	The General Path must not require the core to always produce rich score or metadata objects.
	5.	The General Path must not widen core state ownership.

These rules are required because under MLX lazy execution, extra outputs and extra materialization are not free.

⸻

6. Boundary with Advanced Engines

6.1 What remains in the General Path

The General Path retains:
	•	single-request rich generation semantics,
	•	stop and finish shaping,
	•	richer sampling and processor behavior,
	•	richer score shaping,
	•	generation-facing output construction below product surfaces.

6.2 What must be delegated to engines

The following must be delegated upward:
	•	batching strategy,
	•	speculative orchestration,
	•	prompt-cache orchestration,
	•	admission control,
	•	multi-request scheduling,
	•	engine-level reuse strategies.

6.3 Boundary rule

If a concern coordinates multiple requests, multiple generation sessions, or runtime scheduling across sessions, it is not General Path.

If a concern enriches the semantics of one generation flow above the core contract, it belongs in the General Path.

⸻

7. General Path contract

7.1 Input contract

The General Path input contract may include:
	•	a core generation source or core generation result stream,
	•	generation semantics configuration relevant to Layer 2,
	•	stop policy inputs,
	•	score/logprob options,
	•	sampling/processor options relevant to single-request semantics.

It must not require:
	•	server request objects,
	•	queue/runtime scheduling state,
	•	prompt-cache policy state,
	•	multi-request orchestration state.

7.2 Output contract

The General Path output contract may expose:
	•	richer generation-facing outputs,
	•	richer finish reasons,
	•	optional score/logprob structures,
	•	token/step-level semantic results suitable for upper layers.

It must not expose transport- or server-shaped artifacts as canonical Layer 2 outputs.

7.3 Event / semantic shaping contract

The General Path is allowed to construct generation-facing semantic outputs, including richer token/step structures, provided:
	•	they remain below product transport semantics,
	•	they do not force additional work into the core by default.

7.4 Finish reason contract

The General Path owns the richer finish-reason mapping.

The core only emits minimal continuation/finish state.
The General Path maps that into generation-facing finish reasons, including richer distinctions where needed.

⸻

8. Feature grouping

8.1 Essential General Path features

These are part of the canonical Layer 2 role:
	•	stop semantics,
	•	EOS/max-token shaping,
	•	finish-reason mapping,
	•	richer single-request generation output shaping.

8.2 Rich semantic enrichments

These are canonical enrichments but not part of the fast path:
	•	logprobs/top-logprobs,
	•	richer sampling behavior,
	•	penalties/processors,
	•	richer token/step semantic results.

8.3 Future extensions

These are plausible Layer 2 extensions but not frozen here:
	•	additional generation-semantic adapters,
	•	richer semantic formatting for upper layers,
	•	additional per-request semantic transforms that remain below engines and product surfaces.

Any future extension must preserve the layer boundary rules in this document.

⸻

9. Cost discipline

9.1 Consuming the core without contaminating it

The General Path must treat the core as a narrow-cost substrate.

Binding rule:
	•	the General Path may enrich outputs only after the core contract boundary;
	•	it may not require the core to always compute its optional enrichments.

9.2 Avoiding imposed work on the fast path

The General Path must not force the fast path to pay for:
	•	broad score shaping,
	•	broad semantic metadata,
	•	richer per-step structures,
	•	extra host-side materialization,
	•	product-ready event shaping.

9.3 Keeping additional costs visible

The General Path must preserve cost visibility by:
	•	keeping optional enrichments explicit,
	•	keeping materialization explicit,
	•	avoiding hidden host-side conversions in core-adjacent paths,
	•	making richer semantics opt-in rather than silently mandatory.

This is directly aligned with MLX’s documented lazy execution and implicit evaluation rules.

⸻

10. Abstraction policy

10.1 Admitted abstractions

The General Path may use more abstraction than the Performance Core, but only where it improves:
	•	semantic clarity,
	•	feature grouping,
	•	testability,
	•	composability with upper layers.

Allowed abstraction classes include:
	•	narrow semantic policy objects,
	•	finish-reason mappers,
	•	score/logprob shaping structures,
	•	generation-facing event/value types below product surfaces.

10.2 Still-disallowed abstractions

Still too costly or too invasive:
	•	broad universal generation contexts spanning core, engines, and product surfaces,
	•	abstractions that force the core to produce richer outputs by default,
	•	wrappers that obscure evaluation/materialization points,
	•	product transport objects introduced at Layer 2.

10.3 Acceptance rule

An abstraction in the General Path is acceptable only if:
	1.	it does not widen the core contract,
	2.	it does not hide cost-bearing materialization,
	3.	it isolates genuine Layer 2 semantics,
	4.	it does not absorb Advanced Engines or Product Surface concerns.

⸻

11. Architectural invariants

The following invariants are non-negotiable for the General Path:
	1.	The General Path is Layer 2 and remains above the core.
	2.	The General Path consumes the core; it does not redefine it.
	3.	The General Path owns rich single-request generation semantics.
	4.	The General Path does not own batching, speculative orchestration, or prompt-cache orchestration.
	5.	The General Path does not own transport or server semantics.
	6.	The General Path may enrich outputs only after the core boundary.
	7.	The General Path must not force the fast path to pay for broad semantics.
	8.	The General Path must preserve visibility of added materialization and shaping costs.
	9.	The General Path must remain separable from Advanced Engines and Product Surfaces.
	10.	Richer feature breadth must not be expressed by widening Layer 1.

⸻

12. Open questions deferred to later specs

12.1 Deferred to Advanced Engines specs

The following are explicitly deferred:
	•	batching integration strategy,
	•	speculative integration strategy,
	•	prompt-cache orchestration and reuse semantics,
	•	multi-request scheduling interaction with rich single-request semantics.

12.2 Deferred to Product Surfaces Spec

The following are explicitly deferred:
	•	server-facing output forms,
	•	CLI-facing output forms,
	•	transport framing,
	•	API compatibility objects,
	•	observability payload shaping.

12.3 Deferred to the Implementation Plan

The following are explicitly deferred:
	•	exact module/file decomposition,
	•	migration mechanics from old generate() surfaces,
	•	compatibility shims,
	•	incremental rollout strategy.

12.4 Deferred to later benchmarking detail

The following remain benchmarked later rather than frozen here:
	•	cost profile of optional General Path enrichments,
	•	exact inclusion of Layer 2 features in benchmark subclasses,
	•	performance tradeoffs for score/logprob materialization.
