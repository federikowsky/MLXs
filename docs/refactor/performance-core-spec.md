MLXs — Performance Core Spec

Status: canonical subsystem specification
Role: defines the sovereign core contract of the MLXs refactor
Upstream inputs:
	•	MLXs — Vision / Architectural Foundations
	•	MLXs Research Dossier

This document defines the Performance Core as a technical subsystem. It does not define benchmark methodology in detail, subsystem implementation, or product-facing behavior.

⸻

1. Purpose of the Performance Core

The Performance Core is the sovereign center of the new MLXs architecture.

Its purpose is to own the minimal runtime required to transform prompt state into generated token state under the real execution model of MLX: lazy graph construction, explicit and implicit evaluation boundaries, explicit stream discipline, and host-side materialization costs. The core exists because, in MLX, graph construction and graph evaluation are separate phases, and operations such as eval(), item(), printing arrays, or converting arrays to NumPy materially change execution behavior.  ￼

The Performance Core is sovereign because all higher-level behavior depends on it, while the core must remain independent of higher-level semantics. It is the only subsystem allowed to define:
	•	the primary execution boundary model,
	•	the primary cache ownership model,
	•	the primary cost-visibility model,
	•	the minimal token-generation contract.

It is not the place where feature breadth is expressed. It is the place where execution discipline is fixed.

⸻

2. Core responsibilities

The Performance Core owns only the following responsibilities.

2.1 Prefill

The core owns prompt-state ingestion into runtime state:
	•	processing input token state for generation setup,
	•	populating active-request generation state,
	•	doing so under explicit execution-boundary discipline.

The core does not own prompt formatting, chat-template shaping, or user-facing request semantics.

2.2 Decode-step runtime

The core owns the runtime contract for a decode step:
	•	advancing generation state,
	•	producing the next-step token-generation result,
	•	exposing that result under an explicit output contract,
	•	preserving cost visibility around graph construction and graph evaluation.

Because MLX is lazy and only evaluates at explicit or implicit boundaries, the decode-step runtime is a primary architectural unit, not a helper function.  ￼

2.3 Cache ownership

The core owns the active-request cache during generation:
	•	creation or adoption of active cache state,
	•	mutation of active cache state,
	•	internal rules for cache-state continuity during generation,
	•	controlled export of final or explicitly requested state.

The core does not own higher-level cache reuse policy across requests.

2.4 Eval / sync discipline

The core owns the rules governing:
	•	where evaluation is allowed,
	•	how synchronization is triggered,
	•	when host-side materialization is permitted,
	•	how these costs remain visible.

This follows directly from MLX’s execution model, where eval() and implicit materialization points are execution boundaries with cost.  ￼

2.5 Minimal token-selection contract

The core owns the minimum contract necessary to advance generation:
	•	a token-selection result,
	•	a finish/continuation signal in minimal form,
	•	minimal metadata only when required by the core contract.

This contract must remain smaller than the general generation semantics contract.

⸻

3. Core non-responsibilities

The Performance Core must not own, shape, or understand the following beyond what is strictly required for its boundary contracts.

3.1 Product semantics

The core must not own:
	•	HTTP request/response semantics,
	•	OpenAI-compatible payload shaping,
	•	SSE framing,
	•	CLI interaction semantics,
	•	user-facing observability payloads.

3.2 Rich event shaping

The core must not construct rich per-token product objects such as:
	•	user-facing token events,
	•	server-facing stream events,
	•	tool-call events,
	•	product-facing logging envelopes.

Those belong above the core.

3.3 Advanced orchestration

The core must not own:
	•	continuous batching policy,
	•	speculative decoding orchestration,
	•	prompt-cache admission and eviction policy,
	•	queueing and backpressure policy,
	•	advanced scheduler strategy.

3.4 Server policy

The core must not know:
	•	request timeout policy,
	•	queue-capacity policy,
	•	rejection strategy,
	•	worker model,
	•	API-versioning behavior.

3.5 Broad feature semantics

The core must not be widened by default to carry:
	•	broad logprob shaping,
	•	server-level stop semantics,
	•	tool-calling parsing,
	•	advanced product-layer metadata.

These may consume the core contract from above but must not define it.

⸻

4. Core boundaries

4.1 Input boundary

The input boundary of the core must admit only what the core truly requires:
	•	active model/runtime handle,
	•	input token state or equivalent generation-ready input representation,
	•	active cache state,
	•	narrowly-scoped execution policy,
	•	narrowly-scoped token-selection policy.

The input boundary must not require broad “session context”, “request context”, or “recipe” objects that aggregate unrelated concerns.

4.2 Output boundary

The output boundary of the core must expose:
	•	the next token-generation result,
	•	a continuation/finish signal,
	•	only the minimal metadata needed to preserve correctness and composability.

The output boundary must not force generation of optional rich outputs if the fast path does not require them. This follows from MLX’s documented rule that even unused outputs still have graphs built, which carries cost.  ￼

4.3 State boundary

The state boundary of the core includes:
	•	active mutable generation state,
	•	active cache state,
	•	any minimal state required for token advancement,
	•	explicitly owned state relevant to execution discipline.

The state boundary excludes:
	•	product-level request state,
	•	server scheduling state,
	•	cross-request cache policy state,
	•	rich historical metadata not required to advance generation.

4.4 Layer boundary

The core is Layer 1 in the official architecture.
It may be consumed by upper layers, but must not depend on them.

Binding rule:
	•	upper layers may enrich or orchestrate the core,
	•	upper layers may not redefine the core’s execution model.

⸻

5. Core state model

5.1 Cache ownership

During active generation, the core owns the mutable cache state for that generation.

Binding rules:
	•	there is one active owner of mutable cache state during a generation flow;
	•	upper layers may not maintain independent mutable mirrors of the same runtime cache;
	•	no per-step export/import cycle is allowed between layers.

5.2 Ownership of mutable runtime state

Any mutable runtime state required to advance generation belongs to the core while generation is active.

This includes:
	•	active decode position/state,
	•	active cache mutation state,
	•	any minimal state directly coupled to token advancement.

It excludes:
	•	user-facing semantic history,
	•	product-layer event history,
	•	orchestration state not required for the active generation step.

5.3 Rules on duplication

The core forbids accidental duplication of active mutable runtime state.

Allowed:
	•	immutable snapshots explicitly requested at layer boundaries,
	•	exported terminal state,
	•	diagnostics-only exports outside hot-path contracts.

Disallowed:
	•	mirrored mutable cache representations across adjacent layers,
	•	duplicated active decode state for convenience,
	•	adapters that repeatedly translate equivalent runtime state structures.

5.4 Snapshot, export, and import rules

Snapshots are permitted only under explicit boundary contracts.

Export/import must follow these rules:
	•	export is controlled and explicit;
	•	import must not be part of the per-step fast path unless the core spec is later revised with evidence;
	•	snapshotting must not redefine cache ownership during active generation.

⸻

6. Execution boundary policy

6.1 Eval boundaries

The core must treat evaluation boundaries as architectural boundaries.

Since MLX evaluates lazily and only computes when needed, the core must define where graph evaluation is permitted and where it is forbidden. mx.eval() is an explicit evaluation boundary; item(), printing arrays, and NumPy conversion are implicit ones.  ￼

Binding rule:
	•	no evaluation boundary is “incidental” inside the core.

6.2 item() and materialization rules

The core must treat scalar extraction and host conversion as explicit cost-bearing actions.

Binding rules:
	•	item() is allowed only where explicitly justified by the core contract;
	•	printing MLX arrays is forbidden inside the core;
	•	NumPy conversion is forbidden inside the core fast path;
	•	other host-side materialization must be explicit and contract-visible.

6.3 Sync policy

The core must have an explicit synchronization policy, not an incidental one.

Binding rules:
	•	synchronization points must be enumerable,
	•	synchronization must be attributable to the core contract,
	•	synchronization must not be hidden inside rich objects, logging, or adapter layers.

The exact placement strategy is deferred to the Benchmark Protocol and later validation, but the requirement for explicitness is fixed here.

6.4 Visibility of costs

The core must preserve visibility of:
	•	graph construction cost,
	•	evaluation cost,
	•	synchronization cost,
	•	host-side materialization cost.

Therefore, the core must not hide execution boundaries behind broad semantic interfaces.

⸻

7. Stream policy at spec level

This document does not freeze the final stream policy. It fixes the guarantees the core must satisfy regardless of the final policy selection.

7.1 Stream ownership guarantee

The core must make stream ownership explicit.

MLX operations, including random number generation, are stream-scoped when a stream is supplied; otherwise they run on the default stream of the default device.  ￼

Binding consequence:
	•	the core must not rely on ambiguous stream behavior hidden inside upper-layer abstractions.

7.2 Sampling / RNG relation

Because RNG is stream-scoped in MLX, the core must treat token-selection semantics and stream discipline as related concerns, not independent ones.  ￼

Binding rule:
	•	if the core performs or hosts token-selection behavior that depends on RNG, that behavior must remain compatible with the core’s stream discipline.

7.3 Minimum guarantee before final policy freeze

Before later documents freeze single-stream or multi-stream strategy, the core must guarantee:
	•	it can operate under a coherent single ownership model for execution stream(s),
	•	it does not require product-layer knowledge to maintain stream correctness,
	•	it does not encode assumptions that make later stream-policy refinement impossible.

⸻

8. Core contracts

8.1 Input contract

The core input contract must be:
	•	narrow,
	•	execution-relevant,
	•	free of product semantics,
	•	free of orchestration-only state.

The input contract may include:
	•	model/runtime handle,
	•	active generation-ready token state,
	•	active cache handle,
	•	core execution policy,
	•	minimal token-selection policy.

It must not require:
	•	server request objects,
	•	user-facing generation config objects that bundle unrelated options,
	•	broad lifecycle or orchestration contexts.

8.2 Output contract

The core output contract must expose only:
	•	token-generation result,
	•	minimal finish/continuation state,
	•	optional minimal metadata when required by the core contract.

The output contract must not require broad event materialization or broad semantic packaging.

8.3 Finish / stop signal contract

The core must expose a minimal finish/stop signal contract.

This contract must indicate only what the core is entitled to indicate:
	•	whether generation should continue,
	•	whether a terminal condition has been reached according to the core boundary.

Broader stop-sequence semantics and richer finish-reason shaping are not core responsibilities unless later documents prove that a subset must live in Layer 1.

8.4 Minimal metadata contract

Metadata inside the core contract is allowed only if it is:
	•	necessary for correctness,
	•	necessary for composability with upper layers,
	•	or necessary to preserve cost visibility.

Optional product-facing metadata is excluded.

⸻

9. Fast path relation

The fast path traverses the core as its narrowest and most disciplined execution route.

For the fast path to be possible, the core must preserve these properties:
	•	narrow input contract,
	•	narrow output contract,
	•	no required rich output shaping,
	•	singular cache ownership,
	•	explicit evaluation discipline,
	•	explicit stream discipline,
	•	no forced product-layer semantics.

The core is therefore the enabling substrate of the fast path, but this document does not yet freeze:
	•	the exact benchmark workload class,
	•	the exact fast-path feature slice,
	•	the exact benchmark normalization versus mlx-lm.

Those belong to later documents.

⸻

10. Compile compatibility constraints

The Performance Core is not compile-first. However, it must not foreclose a valid future compile seam.

10.1 Constraints derived from MLX compile model

Official MLX compile constraints include:
	•	compiled functions are intended to be pure,
	•	hidden side effects are unsafe,
	•	the first call can be expensive,
	•	compiled functions are cached,
	•	shape, dtype, dimensionality, and input-count changes can trigger recompilation,
	•	implicit state must be exposed via explicit capture/input mechanisms if it is expected to vary,
	•	RNG state must be explicitly included when relevant.  ￼

10.2 Required preservation properties

Therefore the core must preserve, at minimum:
	•	the possibility of a narrow, explicit step contract,
	•	singular ownership of mutable runtime state,
	•	explicit state boundaries,
	•	explicit execution boundaries,
	•	avoidance of hidden product-layer side effects in the step contract.

10.3 What this does not imply

This does not imply:
	•	compile-first architecture,
	•	immediate adoption of compile in the core,
	•	assumption that the full decode core is compile-compatible.

It implies only that the core must not be designed in a way that makes any future compile seam impossible by construction.

⸻

11. Abstraction policy inside the core

11.1 What is permitted

Permitted abstractions inside the core are limited to those that:
	•	isolate a real responsibility,
	•	preserve narrow contracts,
	•	do not hide evaluation or synchronization cost,
	•	do not widen per-step state transit,
	•	improve local reasoning or testability at negligible hot-path cost.

Examples of potentially admissible categories:
	•	narrow value/control objects,
	•	sharply bounded internal protocols,
	•	immutable configuration carriers that do not cross every step unnecessarily.

11.2 What is forbidden

Forbidden abstractions include:
	•	broad “recipe”, “context”, or “session” containers that mix unrelated concerns,
	•	wrappers over arrays/cache that add dispatch without isolating meaningful responsibility,
	•	adapter chains that convert between nearly equivalent runtime representations,
	•	class hierarchies introduced only for conceptual neatness,
	•	abstractions that make evaluation/materialization cost non-obvious.

11.3 Acceptance criteria for core abstractions

Any abstraction proposed inside the core must answer all of the following:
	1.	What single responsibility does it isolate?
	2.	What runtime cost does it introduce?
	3.	Does it make evaluation/sync/materialization less visible?
	4.	Can the same result be achieved with a narrower data contract?
	5.	Does it preserve future compile compatibility rather than reduce it?

If these answers are weak, the abstraction is rejected.

⸻

12. Architectural invariants of the core

The following invariants are non-negotiable:
	1.	The Performance Core is the sovereign runtime center of MLXs.
	2.	The core owns active mutable generation state during generation.
	3.	The core owns active mutable cache state during generation.
	4.	Evaluation boundaries are explicit architectural boundaries.
	5.	Implicit host materialization is forbidden unless explicitly justified by contract.
	6.	Product semantics must not enter the core contract.
	7.	Rich event shaping must not enter the core contract.
	8.	Advanced orchestration must not redefine the core contract.
	9.	The fast path must not pay for broad general-path outputs by default.
	10.	Core abstractions are governed by runtime cost and boundary clarity.
	11.	The core must preserve the possibility of a future compile seam without becoming compile-first.
	12.	The core must preserve stream correctness without relying on upper-layer semantics.

⸻

13. Open questions intentionally left to later documents

13.1 Deferred to the Benchmark Protocol

The following are intentionally not frozen here:
	•	exact fast-path workload definition,
	•	exact evaluation-boundary benchmarking methodology,
	•	exact synchronization strategy benchmarking,
	•	exact mlx-lm workload normalization rules,
	•	exact stream-policy performance comparison,
	•	exact compile warmup/steady-state methodology.

13.2 Deferred to General Path Spec

The following are intentionally not frozen here:
	•	broader stop semantics,
	•	rich event shaping,
	•	logprob/top-logprob semantics,
	•	richer sampling and penalty semantics,
	•	mapping from core outputs into full generation semantics.

13.3 Deferred to Advanced Engines Specs

The following are intentionally not frozen here:
	•	batching strategy,
	•	speculative orchestration,
	•	prompt-cache orchestration,
	•	scheduler and admission logic,
	•	advanced multi-request runtime policies.

13.4 Deferred to implementation planning

The following remain out of scope here:
	•	concrete module/file structure,
	•	implementation sequencing inside the repo,
	•	migration mechanics,
	•	code-level design choices.
