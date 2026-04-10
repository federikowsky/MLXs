MLXs — Repo Baseline Audit

Status: decision audit
Role: determine the best code baseline from which to restart the refactor
Upstream criteria:
	•	MLXs — Vision / Architectural Foundations
	•	MLXs Research Dossier
	•	MLXs — Performance Core Spec
	•	MLXs — Benchmark Protocol

Scope note

This audit is based on repository evidence already surfaced in prior work and discussion history, not on a fresh full file-by-file inspection in this chat. It is sufficient to decide the recommended baseline direction, the reuse/discard classification, and the main contamination/coupling risks.
It is not sufficient to name an exact commit hash with full certainty; that requires a final live repo verification pass.

⸻

1. Purpose of the audit

This audit exists to unlock one decision:

what real codebase baseline should serve as the restart point for the MLXs refactor.

It must answer, rigorously enough for design governance:
	•	what should be preserved,
	•	what should be extracted,
	•	what should be rebuilt,
	•	what should be discarded,
	•	and which current code paths are structurally incompatible with the new performance-core-first architecture.

This is necessary because the new architecture is not a continuation of the old “single evolving decode engine” direction. It is a controlled restart around a sovereign Performance Core and explicit fast/general path separation.

⸻

2. Audit criteria

The audit uses these criteria.

2.1 Alignment with the Performance Core Spec

The preferred baseline must:
	•	keep the core contract narrow,
	•	preserve explicit state ownership,
	•	avoid hidden evaluation/materialization,
	•	avoid forcing broad outputs through the hot path,
	•	preserve possible future compile compatibility without compile-first design.

2.2 Alignment with layer boundaries

The preferred baseline must make it feasible to separate:
	•	Performance Core,
	•	Generation Semantics,
	•	Advanced Engines,
	•	Product Surfaces.

Any code that structurally collapses these layers counts against the baseline.

2.3 Hot-path contamination

The preferred baseline should minimize:
	•	semantics inside token-step runtime,
	•	product shaping inside generation loops,
	•	broad option handling in the hot path,
	•	compile-specific scaffolding embedded in general runtime flow.

2.4 Protocol contamination

The preferred baseline should avoid broad protocol surfaces that mix:
	•	minimal generation contracts,
	•	rich generation semantics,
	•	orchestration state,
	•	product-facing metadata.

2.5 State duplication

The preferred baseline should minimize:
	•	duplicated active generation state,
	•	cache mirrors,
	•	export/import cycles,
	•	deep-copy-based runtime handoff in core-adjacent paths.

2.6 Costly or unnecessary abstractions

The preferred baseline should avoid:
	•	broad context/recipe/session objects crossing hot code,
	•	wrappers around runtime arrays/cache with little local value,
	•	abstractions that widen step contracts or obscure costs.

2.7 Call-graph reasoning quality

The preferred baseline should offer:
	•	a short, inspectable call graph for generate/decode,
	•	low ambiguity in ownership,
	•	minimal hidden dispatch,
	•	minimal branching accumulated from old experiments.

2.8 Real reusability vs apparent reusability

The audit distinguishes:
	•	code that is genuinely reusable under the new architecture,
	•	code that appears reusable but carries the wrong contracts,
	•	code whose sunk-cost footprint is larger than its future value.

⸻

3. Candidate baselines

3.1 Candidate A — Current late runtime branch with eager-first slice present

Representative evidence from prior work:
	•	src/mlxs/generate/eager_greedy.py exists as a materially distinct slice,
	•	public dispatch into that slice was added in src/mlxs/generate/__init__.py,
	•	compile-centered and resident-state work left significant historical complexity around the generate path.

Advantages
	•	contains the first real nucleus aligned with the new direction: a distinct eager-first greedy slice;
	•	already has some separation signal between narrow path and broad path;
	•	preserves recent benchmark harness evolution and recent correctness work.

Disadvantages
	•	still carries historical contamination from:
	•	compile-centered rebuild phases,
	•	public-path broadening,
	•	generalized generate entry logic,
	•	multi-wave experimental assumptions;
	•	the current generate surface is almost certainly too entangled to serve as the clean root of the new Performance Core;
	•	high risk of salvaging the wrong contracts because some contamination is now normalized in the branch structure.

Judgment

Useful as an evidence source and donor of selected ideas.
Poor as the primary baseline for the core rewrite.

⸻

3.2 Candidate B — Main / pre-decode-optimization baseline

Meaning: a baseline before the major compile-centered / resident-state / eager-wave layering of generation logic.

Advantages
	•	likely the cleanest available root for reasoning;
	•	lowest probability of carrying experimental protocol contamination;
	•	better match for a true restart where the core is rebuilt intentionally rather than “rescued”.

Disadvantages
	•	may lack recent harness corrections, recent tests, and recent narrow-slice ideas;
	•	may require re-porting some useful improvements from later work;
	•	if too old, may lose unrelated repository improvements that are still valid outside the runtime core.

Judgment

Strong candidate for the runtime restart point.

⸻

3.3 Candidate C — Hybrid baseline

Meaning:
	•	keep the latest stable repository baseline for non-runtime subsystems,
	•	replace the current generate/decode core subtree with a fresh rebuild starting from a cleaner earlier state or from new code.

Advantages
	•	best fit to the new architecture in practice;
	•	avoids throwing away mature subsystems that are not part of the decode-core failure;
	•	limits rewrite scope to the parts most affected by architectural misalignment;
	•	aligns with the layer model: core is rebuilt, upper/lateral subsystems are selectively retained.

Disadvantages
	•	requires discipline to prevent current generate-path contracts from bleeding into the new core;
	•	some interfaces around server, batch, prompt cache, and config will need temporary decoupling work;
	•	can create short-term integration friction because upper layers currently assume the old generate surface.

Judgment

Best overall direction.

⸻

3.4 Candidate D — Continue from current code and refactor in place

Advantages
	•	lowest immediate churn;
	•	least mechanical reshuffling.

Disadvantages
	•	highest risk of preserving the wrong contracts;
	•	highest risk of “incremental cleanup” reintroducing the old single-engine mindset;
	•	strongest sunk-cost trap;
	•	worst fit for performance-core-first architecture.

Judgment

Rejected.

⸻

4. Hot-path contamination map

The current runtime appears contaminated in the following ways.

4.1 Broad public generate entrypoint contamination

The current generate() surface has historically been responsible for:
	•	prompt encoding,
	•	sampler/logits-processor resolution,
	•	stop semantics,
	•	cache wiring,
	•	prefill orchestration,
	•	decode-loop selection,
	•	optional final cache export,
	•	compile flags,
	•	logprob-related behavior.

This is too broad for the new architecture.
It conflates:
	•	core runtime concerns,
	•	semantics-layer concerns,
	•	upper-layer orchestration concerns.

4.2 General decode-loop contamination

The general decode path appears to have accumulated support pressure from:
	•	greedy and sampling modes,
	•	logprobs,
	•	logits processors,
	•	quantized-KV transitions,
	•	compile toggles,
	•	clearing policy,
	•	output packaging.

That breadth is exactly what the new Performance Core must not inherit by default.

4.3 Public wrapper contamination

There is evidence that public wrappers and outer generation layers currently do more than simple boundary conversion:
	•	they influence control flow,
	•	they shape per-token semantics,
	•	they may still affect what the loop must produce.

This means the current boundary between core runtime and generation semantics is too porous.

4.4 Prompt-cache contamination of generation contracts

From prior evidence, prompt-cache handling stores and returns deep copies of full per-layer cache lists. That is acceptable as an upper-layer mechanism, but it is a poor shape to allow near the core contract. It encourages duplicated state and blurred cache ownership.

4.5 Compile contamination

The repository history shows repeated attempts to make compile a central architectural dimension of the decode path. Even where those experiments were later narrowed or rejected, they likely left:
	•	compile-centric interfaces,
	•	compile flags in broad public paths,
	•	state assumptions shaped by compile concerns rather than core-contract discipline.

4.6 Product-surface contamination risk

Server routes and server dependencies call into generate() directly enough that the current generation API likely still reflects server/product needs more than core-runtime needs.

⸻

5. State ownership audit

5.1 Active cache ownership

Current ownership appears too diffuse.

Known issues from prior evidence:
	•	core generation mutates active cache,
	•	prompt-cache logic deep-copies cache state for reuse,
	•	final-cache export can expose core-owned cache outward,
	•	multiple cache protocol variants exist across architectures (KVCache, rotating, quantized, chunked, CacheList, ArraysCache).

This is not wrong in itself, but it means the current generate contract is too exposed to cache heterogeneity and cross-layer state concerns.

5.2 Duplication of state

Known duplication risks:
	•	prompt-cache deep copies,
	•	exported final cache references,
	•	potentially duplicated semantic/runtime state between generation wrappers and runtime loops,
	•	per-feature handling that forces the runtime to carry state it should not own.

5.3 Export/import patterns

The historic compile-centered line introduced or explored multiple forms of state export/import and rematerialization. Even when those paths are no longer active, their influence on interfaces is a contamination signal.

Under the new architecture, per-step or per-mode export/import of runtime state is a negative signal unless strictly justified.

5.4 Adapter and mirror structures

The existence of broad cache protocols and multiple cache families is not itself a defect, but the current generate surface likely acts as a too-broad adapter over all of them. That weakens singular ownership and makes the core contract too generic.

⸻

6. Protocol and abstraction audit

6.1 Current protocol shape

The current generation area appears to rely on broad entry contracts that aggregate:
	•	generation options,
	•	cache choices,
	•	stop/logprob behavior,
	•	compile/clear-cache knobs,
	•	optional outputs.

This is wider than the new Performance Core allows.

6.2 Over-wide interfaces

The following kinds of interface appear problematic:
	•	a universal generate() contract expected to serve both narrow fast path and broad general path;
	•	broad “options” objects whose fields span multiple layers;
	•	step-function contracts that attempt to serve eager, compile, logprobs, and richer semantics at once.

6.3 Context / recipe / session objects

The historical analysis repeatedly referenced recipe/plan/context-style patterns around generation. Regardless of the exact class names, this style is a warning sign for the new core:
	•	it centralizes too many concerns,
	•	it hides cost-bearing decisions,
	•	it encourages broad transit objects through the hot path.

6.4 Abstractions obstructing the new core

Most suspect abstraction categories:
	•	universal decode-step builders,
	•	broad generation wrapper contracts,
	•	context objects crossing every step,
	•	cache-generic logic embedded in the main hot path,
	•	compile-oriented indirection embedded in the same call graph as eager execution.

These do not all need deletion at the repo level, but they should not shape the new core.

⸻

7. Reusability classification

7.1 Keep

Subsystems or areas likely reusable largely as-is:
	•	model loading and registry infrastructure, unless tied too tightly to old generate contracts;
	•	model architecture implementations and shared layers, as they are not the primary locus of the refactor;
	•	cache implementations as substrate components, though not their current broad coupling into generate contracts;
	•	converter/model-format work;
	•	tokenization and model-loading support code;
	•	broad non-runtime utilities and much of config/observability infrastructure outside hot paths.

7.2 Keep with extraction / refactor

Useful, but only after decoupling:
	•	chunked prefill logic;
	•	the narrow eager-first slice ideas in generate/eager_greedy.py;
	•	benchmark harness assets and result corpus structure;
	•	selected server transport shell pieces, but only above a redefined runtime boundary;
	•	selected cache protocol abstractions, if narrowed by layer.

7.3 Rework heavily

Areas likely too contaminated to keep in current form:
	•	src/mlxs/generate/__init__.py as the public runtime entry shape;
	•	general decode loop / broad decode path;
	•	compile-related generation integration;
	•	prompt-cache integration around generate contracts;
	•	batching code if it assumes the current generate shape;
	•	any broad cross-layer generation options objects.

7.4 Discard

Discard as architectural baselines for the new core:
	•	the old single evolving decode-engine line of thought as a design basis;
	•	any resident-state experimental assumptions that shaped runtime interfaces;
	•	any compile-centered step contracts that require broad mutable state shaping inside the core;
	•	any broad per-step structures created only to support “one engine for all modes”.

⸻

8. Coupling hotspots

8.1 Generate ↔ Server

The server appears to call into generate surfaces that are too broad and too semantically loaded. This is a major hotspot because product-surface requirements likely shaped the current runtime interface.

8.2 Generate ↔ Prompt cache

Prompt-cache reuse sits too close to runtime state shape today. That makes cache reuse concerns more invasive than they should be for the new core.

8.3 Generate ↔ Batch

If batching depends on current generate contracts rather than a lower-level runtime substrate, it is coupled to the wrong layer boundary.

8.4 Generate ↔ Compile

Compile-related integration is a hotspot because compile concerns historically influenced the shape of runtime contracts instead of remaining subordinate.

8.5 Generate ↔ Config

Broad option/config objects likely inject too many knobs directly into the runtime path. This widens the core interface and makes reasoning harder.

8.6 Cache protocol breadth ↔ Core reasoning

Supporting many cache families is legitimate, but letting that support shape the same minimal decode contract is a hotspot. The new architecture needs a narrower core-facing cache contract.

⸻

9. Risk assessment

9.1 Risk of restarting from the wrong base

If the refactor starts from the current late runtime branch as the architectural base:
	•	old contracts may be preserved by inertia,
	•	the new core may inherit the wrong abstractions,
	•	the fast path may become a cleaned-up version of the old general path instead of a true sovereign core.

This is the highest-risk mistake.

9.2 Risk of saving too much

If too much current generate/runtime code is preserved:
	•	state ownership will stay blurred,
	•	protocol contamination will survive,
	•	broad options/context objects will remain “too expensive to remove”,
	•	benchmark classes will be harder to separate cleanly.

9.3 Risk of rewriting too much

If everything is rewritten indiscriminately:
	•	stable non-runtime infrastructure may be lost,
	•	unrelated maturity in loading/models/cache/converter/server shell may be discarded unnecessarily,
	•	schedule risk increases without corresponding architectural benefit.

9.4 Correct balance

The least risky path is:
	•	preserve mature non-core subsystems,
	•	rebuild the runtime core and its immediate boundaries decisively,
	•	decouple upper layers from the old generate contract rather than preserving that contract.

⸻

10. Baseline recommendation

Final recommendation

Recommended baseline:
Hybrid baseline (Candidate C)

Precise meaning
	•	use the latest stable repository state as the non-runtime substrate baseline,
	•	but do not use the current generate/decode architecture as the baseline for the new core,
	•	instead, restart the core/runtime subtree from a cleaner earlier state or from fresh code informed by the canonical documents,
	•	selectively port only the narrow reusable ideas from the late branch.

Technical rationale

This recommendation best satisfies the canonical criteria:
	•	it avoids preserving contaminated runtime contracts;
	•	it retains mature, non-problematic repository value outside the core;
	•	it aligns with the layered architecture by rebuilding Layer 1 without forcing rewrites of Layers 2–4 where not necessary;
	•	it minimizes sunk-cost bias while also avoiding unnecessary full-repo churn.

Operational baseline recommendation

For the actual restart point, the preferred code baseline should be:

latest stable mainline for non-runtime subsystems + fresh runtime-core subtree replacement

If an exact earlier commit is needed as a donor for the runtime subtree, it should be chosen from the pre-compile-centered / pre-resident-state / pre-eager-wave accumulation era, subject to final live verification.

What is explicitly not recommended
	•	continuing from the current late runtime branch as the architectural root;
	•	refactoring in place from the current broad generate path;
	•	preserving the current public generate contract as the new core boundary.

⸻

11. Immediate consequences

This baseline choice implies the following.

11.1 The next subsystem specs become higher priority

After this audit, the most important next specs are:
	1.	General Path Spec
Because a fresh core implies the old public generate contract will not survive unchanged.
	2.	Advanced Engines Spec
Because batching, speculative decoding, and prompt-cache orchestration must be reattached above the new core rather than baked into it.
	3.	Product Surfaces Spec
Because server and CLI must consume a different lower boundary than they do today.

11.2 The future implementation plan must be runtime-subtree-first

The implementation plan, when written later, must assume:
	•	Layer 1 is rebuilt first,
	•	existing generate-facing integrations are temporarily adapters or compatibility shells at most,
	•	upper layers are reattached after the core boundary is stable.

11.3 The benchmark harness can be retained as a strategic asset

The benchmark harness and historical result corpus remain valuable, but they must now be used against:
	•	the new core boundary,
	•	the canonical benchmark classes,
	•	and the corrected fairness rules.

11.4 Final live inspection is still required

A short final repo verification pass is still needed to:
	•	choose the exact donor commit for runtime restart if required,
	•	confirm which current files are free of hidden coupling,
	•	map exact module replacement boundaries.

That live inspection is a verification step, not a reason to reopen the architectural decision.