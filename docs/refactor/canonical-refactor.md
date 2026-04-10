MLXs — Vision / Architectural Foundations

Status: canonical refactor foundation
Purpose: root document for the MLXs rebuild; governs the documents that follow
Input basis: current architectural direction and the Research Dossier grounded in official MLX documentation and current mlx-lm public surfaces. MLX facts used here inherit from the verified dossier basis: lazy execution, explicit and implicit evaluation boundaries, stream-scoped execution including RNG, compile purity/state/shape constraints, and the current comparable mlx-lm generation path and server/runtime surface.

⸻

1. Purpose of the refactor

MLXs is being refounded because the prior direction accumulated too much structural complexity in the inference core relative to the value it produced. The rebuild is not a feature refresh. It is a correction of the library’s architectural center of gravity.

The structural problems being corrected are these:
	•	the decode/runtime core was not sufficiently separated from broader generation semantics and product concerns;
	•	performance-sensitive execution concerns were not isolated strongly enough from feature breadth;
	•	abstractions were allowed to widen hot-path contracts and hide runtime costs that are material under MLX’s lazy execution model;
	•	compile-related reasoning and general-path requirements were allowed to influence core structure too early;
	•	the library lacked a single canonical architectural root document that ordered priorities and constrained downstream design.

The refactor exists to establish a new foundation in which the core runtime matches MLX’s real execution model more directly, preserves cost visibility, and supports broader functionality without allowing that breadth to govern the performance-critical path. MLX’s documented lazy graph model, explicit and implicit evaluation boundaries, stream semantics, and compile constraints make this restructuring necessary rather than optional.

⸻

2. Design priorities

The official priority order of the refactor is:
	1.	Performance-core integrity
The core inference path must be structurally minimal, cost-legible, and aligned with MLX runtime behavior.
	2.	Correctness under the true MLX execution model
Evaluation boundaries, stream usage, host materialization, and state ownership must be explicit and correct.
	3.	Architectural rigor
Layers, paths, and interfaces must reflect real responsibilities and real costs.
	4.	Benchmarkable competitiveness against mlx-lm
The system must support fair, reproducible comparison against the current mlx-lm baseline on properly defined workload classes. The current mlx-lm generation path already defines a meaningful comparable substrate: explicit generation stream, mx.async_eval, scalar materialization, and periodic mx.clear_cache().
	5.	General-path correctness and breadth
Richer generation semantics and broader runtime features must remain available, but not at the expense of the core.
	6.	Product configurability and surface completeness
Config layering, server behavior, observability, and wider product affordances are important, but they are downstream of the performance-core design.

This ordering is binding for all subsequent documents.

⸻

3. Architectural worldview

3.1 Performance-core-first

MLXs is defined around a sovereign performance core. The performance core is the primary architectural unit of the system. It is not one subsystem among many. It is the reference point against which other layers are evaluated.

This follows directly from the MLX execution model: because MLX is lazy and actual computation happens only at evaluation boundaries, the shape of the runtime core and its boundary discipline materially affect performance and correctness.

3.2 Dual-path by design

MLXs is dual-path by design:
	•	Fast path: competitive path intended to minimize work and maximize comparability on benchmarkable generation workloads.
	•	General path: broader path intended to preserve correctness and feature breadth for richer workloads.

This is not a temporary simplification. It is a permanent architectural principle. The fast path is not a narrowed mode of a universal engine.

3.3 Eager-first, compile-subordinated

The rebuild treats eager execution as the primary architectural baseline. Compile is recognized as a powerful optimization tool, but it is subordinate to a correct, minimal, benchmarked eager core.

This is required by the documented MLX compile model: compile is effective when functions are pure, repeatedly reused, state-disciplined, and shape/type stable enough to avoid pathological recompilation or unsafe hidden-state behavior.

3.4 Abstraction discipline

Abstractions are allowed only when they improve:
	•	responsibility separation,
	•	local reasoning,
	•	testability,
	•	or controlled extensibility,

without widening the hot path or obscuring material runtime costs.

Abstraction is therefore constrained by runtime cost, not only by conceptual neatness.

3.5 MLX-aware and hardware-aware design

The new MLXs is explicitly MLX-aware and hardware-aware. This means:
	•	evaluation boundaries are treated as architectural boundaries;
	•	stream usage is treated as a runtime ownership concern;
	•	host-side materialization is treated as explicit cost;
	•	native MLX fast primitives are treated as trusted substrate candidates in the core;
	•	graph size, graph shape, synchronization placement, and data movement are first-class design concerns.

This worldview is grounded in official MLX behavior, not stylistic preference.

⸻

4. System structure

The official layer model of the new MLXs is:

Layer 1 — Performance Core

Responsibilities
	•	prefill and decode runtime primitives,
	•	active-request cache ownership,
	•	evaluation and synchronization discipline,
	•	minimal token-selection contract,
	•	strict core runtime control flow.

Does not own
	•	server semantics,
	•	rich event shaping,
	•	prompt-cache policy,
	•	advanced scheduling,
	•	tool calling,
	•	product-facing API shaping.

Layer 2 — Generation Semantics

Responsibilities
	•	stop semantics,
	•	richer sampling behavior,
	•	penalties/processors,
	•	logprobs/top-logprobs,
	•	conversion of core outputs into generation-facing semantics.

Constraint
This layer consumes the core. It does not redefine the core.

Layer 3 — Advanced Engines

Responsibilities
	•	batching,
	•	speculative decoding,
	•	prompt-cache orchestration,
	•	runtime scheduling and admission logic for advanced flows.

Constraint
This layer coordinates core invocations and higher-level policies without widening the performance-core contract.

Layer 4 — Product Surfaces

Responsibilities
	•	HTTP/SSE server,
	•	CLI,
	•	config,
	•	observability,
	•	lifecycle management,
	•	broader product-facing integration surfaces.

Constraint
This layer is the composition root. It must remain downstream of the lower layers.

Dependency rules

The dependency rules are:
	•	Layer 1 depends only on trusted substrate, core-local utilities, and explicitly allowed contracts.
	•	Layer 2 depends on Layer 1.
	•	Layer 3 depends on Layers 1 and 2.
	•	Layer 4 depends on all lower layers as the composition root.
	•	Lower layers must not depend on higher layers.
	•	Product-facing semantics must not leak downward into the performance core.
	•	Advanced orchestration must not widen the fast-path loop contract.

These rules are non-negotiable.

⸻

5. Execution model

5.1 Fast path

The fast path is the competitive path. Its purpose is to minimize per-token work, minimize accidental graph growth, minimize host-side materialization, and keep evaluation and stream behavior explicit.

The fast path is expected to begin with a narrow workload slice and stay structurally narrow unless later documents justify widening with evidence.

5.2 General path

The general path supports richer generation and runtime behavior:
	•	broader sampling semantics,
	•	logprobs,
	•	richer penalties/processors,
	•	advanced cache behavior,
	•	advanced orchestration,
	•	wider server/runtime affordances.

The general path is allowed to be more expensive when the workload requires it.

5.3 Relationship between path and layer

The fast path lives primarily in Layer 1 and is enriched minimally by Layer 2 only where required by its defined workload class.

The general path spans Layers 1–3 and is surfaced by Layer 4.

Path is therefore orthogonal to layering:
	•	the fast path is a constrained traversal through the architecture;
	•	the general path is a broader traversal through the architecture.

This distinction must remain explicit in all downstream documents.

⸻

6. Architectural invariants

The following invariants govern the refactor:
	1.	The performance core is sovereign.
	2.	Evaluation boundaries are explicit architectural boundaries.
	3.	Implicit host materialization is forbidden in the performance core except where explicitly allowed by contract.
MLX documents that item(), printing arrays, and NumPy conversion all force evaluation.
	4.	The fast path must not pay for general-path outputs unless proven negligible.
	5.	The fast path is not a mode of a universal engine.
	6.	Compile is subordinate to a proven eager core.
	7.	State ownership must be singular and visible.
	8.	Abstractions in the core are allowed only when they compress the system rather than expand it.
	9.	Benchmarkability is a design requirement, not a later validation exercise.
	10.	Trusted MLX-native primitives are preferred substrate candidates in the core unless evidence justifies an alternative.
The official MLX fast substrate includes fast SDPA, norms, RoPE, and custom-kernel escape hatches.

⸻

7. Refactor boundaries

7.1 What is in scope

The refactor includes:
	•	redefining the architectural center of MLXs,
	•	rebuilding the inference/runtime core,
	•	redefining path and layer boundaries,
	•	redefining the role of compile within MLXs,
	•	redefining benchmarkable workload classes,
	•	restructuring contracts between runtime, semantics, advanced engines, and product surfaces.

7.2 What must not guide the initial design

The following must not drive the initial design:
	•	server-surface breadth,
	•	feature parity in its broadest sense,
	•	backward continuity with prior internal decode-engine designs,
	•	compile-first optimism,
	•	product-surface convenience if it widens the core contract.

7.3 What is deferred

The following are deferred to later documents:
	•	exact benchmark taxonomy and benchmark protocol details,
	•	detailed Performance Core specification,
	•	exact compile seam design,
	•	detailed stream policy selection,
	•	subsystem-level documents for batching, speculative, prompt cache, and server internals,
	•	implementation planning.

These are downstream of the present foundations document.

⸻

8. Implementation order

The correct order of refinement after this document is:
	1.	Performance Core Spec
Defines the core runtime contract, state model, boundary discipline, and core execution responsibilities.
	2.	Benchmark Protocol
Defines workload classes, comparison rules, normalization assumptions against mlx-lm, and validation methodology.
	3.	Repo Baseline Audit
Selects the cleanest restart baseline and identifies what code can or cannot be reused.
	4.	General Path Spec
Defines richer generation semantics above the core.
	5.	Advanced Engines Specs
Covers batching, speculative decoding, prompt-cache orchestration, and related advanced runtime structures.
	6.	Product Surface Specs
Covers server, CLI, observability, configuration, and lifecycle.

This order is intentional: it moves from architectural center to progressively wider surfaces.

⸻

9. Derived document map

The next documents to produce are:

1. MLXs — Performance Core Spec

Defines:
	•	the core runtime boundary,
	•	core input/output contracts,
	•	cache ownership rules,
	•	sync/eval discipline,
	•	eager baseline model,
	•	compile eligibility constraints.

2. MLXs — Benchmark Protocol

Defines:
	•	benchmark workload classes,
	•	competitive baseline rules against mlx-lm,
	•	measurement boundaries,
	•	warmup policy,
	•	feature inclusion/exclusion rules by benchmark class.

3. MLXs — Repo Baseline Audit

Defines:
	•	restart baseline candidates,
	•	reusable code,
	•	discard candidates,
	•	existing coupling hotspots,
	•	migration risks.

4. MLXs — General Path Spec

Defines:
	•	generation semantics above the core,
	•	richer sampling,
	•	stop/logprobs/event shaping contracts.

5. MLXs — Advanced Engines Spec

Defines:
	•	batching,
	•	speculative decoding,
	•	prompt-cache orchestration,
	•	advanced scheduling boundaries.

6. MLXs — Product Surfaces Spec

Defines:
	•	server,
	•	CLI,
	•	configuration,
	•	observability,
	•	lifecycle.

7. MLXs — Implementation Plan

Defines:
	•	execution phases,
	•	migration strategy,
	•	validation order,
	•	rollout structure.

⸻

10. Decision summary

10.1 Decisions already frozen

The following decisions are frozen by this document:
	•	MLXs is being rebuilt around a performance-core-first architecture.
	•	MLXs is dual-path by design.
	•	The rebuild is eager-first, with compile subordinated to a proven eager core.
	•	The official system structure is four-layered:
	1.	Performance Core
	2.	Generation Semantics
	3.	Advanced Engines
	4.	Product Surfaces
	•	Lower layers must not depend on higher layers.
	•	Product and feature breadth must not govern the performance core.
	•	Abstractions in the core are governed by runtime cost and responsibility separation, not by theoretical elegance.
	•	The Research Dossier is the factual substrate for downstream technical documents.

10.2 Decisions still open, but deferred

The following decisions remain open and are intentionally deferred:
	•	exact fast-path workload slice,
	•	exact general-path widening rules,
	•	exact compile seam and compile/default policy,
	•	exact stream policy for the performance core,
	•	exact benchmark workload taxonomy,
	•	exact baseline commit/branch from which to restart implementation,
	•	exact subsystem contracts for batching, speculative, prompt cache, and server wiring.

These are not unresolved because of indecision. They are deferred because this document defines foundations, not subsystem detail.