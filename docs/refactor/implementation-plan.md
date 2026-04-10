MLXs — Implementation Plan

Status: canonical execution plan
Role: defines how to execute the MLXs refactor phase by phase
Upstream inputs:
	•	MLXs — Vision / Architectural Foundations
	•	MLXs Research Dossier
	•	MLXs — Performance Core Spec
	•	MLXs — Benchmark Protocol
	•	MLXs — Repo Baseline Audit
	•	MLXs — General Path Spec
	•	MLXs — Advanced Engines Spec
	•	MLXs — Product Surfaces Spec

⸻

1. Purpose of the Implementation Plan

This document exists to govern execution of the refactor.

It defines:
	•	the order in which the refactor must proceed,
	•	the cut lines between phases,
	•	the dependencies between phases,
	•	the validation model per phase,
	•	the migration strategy from the current codebase,
	•	the criteria for declaring the refactor complete.

It does not redefine architecture, reopen baseline selection, or provide code-level implementation detail.

The plan exists because MLX’s execution model makes boundary discipline materially important: evaluation boundaries, implicit materialization, stream usage, and compile constraints must remain aligned with the architectural specs during execution, not merely in theory.  ￼

⸻

2. Execution principles

2.1 Architecture-first

Implementation must follow the canonical architecture documents, not discover architecture ad hoc through coding.

Binding rule:
	•	if implementation pressure conflicts with the architecture, the conflict must be resolved explicitly before code proceeds.

2.2 Runtime-subtree-first

Execution begins with the runtime core subtree, not with product surfaces and not with broad feature restoration.

Binding rule:
	•	Layer 1 is rebuilt first,
	•	higher layers reattach only after lower-layer contracts are stable.

2.3 Benchmark-governed validation

Progress is not declared by “code exists.”
Progress is declared by:
	•	architectural conformance,
	•	correctness,
	•	benchmark-class validation appropriate to the phase.

The canonical fast-path benchmark governs the first implementation cycle.

2.4 No contract drift

Temporary implementation convenience must not redefine:
	•	Layer 1 contracts,
	•	Layer 2 contracts,
	•	Layer 3 contracts,
	•	Layer 4 as composition root.

Temporary shims are allowed only if they are explicitly transitional and do not become de facto architecture.

2.5 Fresh-core bias

Where there is tension between:
	•	preserving old runtime code,
	•	and preserving the new architecture,

the new architecture wins.

This follows from the Repo Baseline Audit conclusion: latest stable non-runtime substrate plus fresh runtime-core subtree replacement.

2.6 Controlled reattachment

Upper layers must be reattached deliberately and in order:
	•	first Layer 2,
	•	then Layer 3,
	•	then Layer 4.

No phase may “skip upward” by product-wiring around unfinished lower-layer contracts.

⸻

3. Refactor strategy

3.1 General strategy

The refactor strategy is:
	•	retain stable non-runtime subsystems where their contracts remain compatible,
	•	rebuild the runtime core subtree fresh,
	•	reattach upper layers in controlled phases,
	•	use benchmark-governed validation at each major cut line,
	•	remove temporary compatibility structures as soon as their purpose ends.

3.2 Fresh runtime-core subtree replacement

The old generate/decode architecture is not the baseline.
The runtime core is rebuilt as a fresh subtree shaped by:
	•	the Performance Core Spec,
	•	the Benchmark Protocol,
	•	the General Path Spec.

3.3 Retention of stable non-runtime subsystems

Subsystems outside the contaminated runtime center may be retained, including:
	•	model loading and registry infrastructure,
	•	model implementations and shared layers,
	•	cache substrate components,
	•	converter-related code,
	•	broad non-runtime utilities,
subject to interface conformance during reintegration.

3.4 Controlled reattachment of upper layers

Upper layers must reattach only to the new lower-layer contracts.
They may use temporary adapters during transition, but they must not force old contracts back downward.

⸻

4. Phased plan

Phase 0 — Preparation / branch strategy / carve-out

Purpose:
	•	establish execution branch strategy,
	•	define the runtime-core carve-out boundary,
	•	confirm the baseline substrate and the replacement subtree boundaries,
	•	freeze temporary branch discipline.

Phase 1 — Layer 1 implementation

Purpose:
	•	implement the Performance Core as a fresh subtree,
	•	establish the sovereign core contract in code,
	•	prevent carryover of the old generate/decode shape.

Phase 2 — Benchmark alignment and validation

Purpose:
	•	align actual benchmark harness behavior with the canonical Benchmark Protocol,
	•	validate the new Layer 1 against the canonical fast-path benchmark class,
	•	ensure measurement boundaries and fairness rules are correct before higher-layer reattachment.

Phase 3 — Layer 2 implementation

Purpose:
	•	implement the General Path above the new core,
	•	restore rich single-request generation semantics without widening Layer 1.

Phase 4 — Layer 3 implementation

Purpose:
	•	implement Advanced Engines above Layers 1–2,
	•	restore orchestration surfaces without recreating a universal generation engine.

Phase 5 — Layer 4 implementation / reattachment

Purpose:
	•	implement and/or reattach Product Surfaces as the composition root,
	•	expose lower-layer capabilities without deforming them.

Phase 6 — Integration hardening / cleanup / compatibility reduction

Purpose:
	•	remove transitional adapters and legacy compatibility shells,
	•	reduce residual contamination,
	•	harden the integrated system,
	•	validate architectural completeness.

⸻

5. Per-phase goals

Phase 0

Goal

Freeze execution setup and isolate the runtime subtree boundary.

Deliverable
	•	agreed refactor branch structure,
	•	confirmed runtime-core replacement boundary,
	•	confirmed retained non-runtime substrate boundary.

Dependencies
	•	all canonical documents through Product Surfaces Spec,
	•	Repo Baseline Audit.

Exit criteria
	•	no ambiguity about what is being replaced vs retained,
	•	no ambiguity about the first code area to be rebuilt,
	•	no ambiguity about which old contracts are transitional only.

⸻

Phase 1

Goal

Create the new Layer 1 implementation.

Deliverable
	•	fresh Performance Core subtree,
	•	Layer 1 contract represented in code,
	•	no dependency on old universal generate/decode logic.

Dependencies
	•	Performance Core Spec,
	•	Repo Baseline Audit,
	•	Research Dossier.

Exit criteria
	•	Layer 1 exists as a distinct runtime subtree,
	•	active state/cache ownership follows the Layer 1 spec,
	•	old broad generate boundary is not the canonical core boundary anymore,
	•	lower-layer execution discipline is inspectable without product-surface contamination.

⸻

Phase 2

Goal

Validate Layer 1 under the canonical benchmark class and align the harness with the Benchmark Protocol.

Deliverable
	•	canonical Class A benchmark execution against mlx-lm,
	•	reproducible benchmark reports,
	•	validated measurement boundaries,
	•	fairness normalization confirmed.

Dependencies
	•	Layer 1 implementation,
	•	Benchmark Protocol,
	•	Research Dossier.

Exit criteria
	•	benchmark harness reflects canonical workload-class separation,
	•	warmup/reporting/fairness rules are enforced,
	•	Layer 1 correctness is confirmed for the benchmarked path,
	•	no benchmark drift relative to the canonical protocol.

⸻

Phase 3

Goal

Implement Layer 2 without widening Layer 1.

Deliverable
	•	General Path implementation above the core,
	•	rich single-request semantics restored,
	•	no reintroduction of the old universal generate boundary.

Dependencies
	•	stable Layer 1,
	•	validated benchmark setup,
	•	General Path Spec.

Exit criteria
	•	Layer 2 consumes Layer 1 without redefining it,
	•	rich semantics exist independently of Layer 3 and Layer 4,
	•	fast path remains realizable without Layer 2 overhead where not required.

⸻

Phase 4

Goal

Implement Layer 3 as advanced orchestration above Layers 1–2.

Deliverable
	•	batching/speculative/prompt-cache orchestration surfaces implemented or reattached to new lower-layer contracts,
	•	orchestration state isolated in Layer 3.

Dependencies
	•	stable Layer 1,
	•	stable Layer 2,
	•	Advanced Engines Spec.

Exit criteria
	•	orchestration concerns no longer leak into Layer 1 or Layer 2,
	•	multi-request coordination lives in Layer 3,
	•	old mixed generate/orchestration patterns are not retained as canonical paths.

⸻

Phase 5

Goal

Implement/reattach Layer 4 as composition root.

Deliverable
	•	server surface,
	•	CLI surface,
	•	config surface,
	•	observability surface,
	•	lifecycle/management surface,
all consuming the lower layers according to the Product Surfaces Spec.

Dependencies
	•	stable Layers 1–3,
	•	Product Surfaces Spec.

Exit criteria
	•	Layer 4 is the only composition root,
	•	product shaping remains out of lower layers,
	•	compatibility/API concerns are handled as Layer 4 concerns.

⸻

Phase 6

Goal

Harden the integrated system and remove transitional structures.

Deliverable
	•	reduced or removed compatibility shims,
	•	cleanup of legacy paths,
	•	integration validation,
	•	final architecture-conformance pass.

Dependencies
	•	integrated Layers 1–4,
	•	all prior phase validations.

Exit criteria
	•	transitional adapters removed or minimized,
	•	no old runtime boundary remains canonical by accident,
	•	the integrated system is coherent with all canonical documents.

⸻

6. Validation model

6.1 Per-phase validation model

Each phase is validated on four axes:
	•	architectural conformance
	•	correctness
	•	regression checks
	•	benchmark validation where applicable

6.2 Architectural conformance

At each phase:
	•	code must conform to the governing layer spec,
	•	lower-layer contracts must not be widened by upper-layer needs,
	•	temporary shims must be explicitly identified.

6.3 Correctness checks

Correctness checks must validate:
	•	contract integrity,
	•	ownership rules,
	•	semantic fidelity appropriate to the layer,
	•	absence of invalid cross-layer leakage.

6.4 Regression checks

Regression checks must verify:
	•	no reintroduction of discarded legacy boundaries,
	•	no new hidden product-shaped runtime contracts,
	•	no new default overhead imposed on lower layers by higher layers.

6.5 Benchmark validation

Benchmark validation is mandatory beginning in Phase 2 and continues through later phases for relevant benchmark classes.

The canonical fast-path benchmark governs the earliest success criteria because it maps directly to Layer 1 and the first-cycle goal. mlx-lm’s comparable generation path currently includes explicit generation-stream use, mx.async_eval, scalar materialization of token outputs, and periodic mx.clear_cache(), so fairness normalization must remain tied to those real behaviors.  ￼

⸻

7. Migration strategy

7.1 Migration from the old generate/decode asset

Migration proceeds by replacement, not by gradual mutation of the old generate/decode center.

The strategy is:
	•	carve out the old canonical runtime boundary,
	•	introduce the new Layer 1 subtree,
	•	adapt the old repository around it in controlled phases.

7.2 Temporary adapters or shims

Temporary adapters are permitted only to:
	•	keep upper layers usable during staged reattachment,
	•	bridge old product surfaces to new lower-layer contracts,
	•	avoid blocking all progress behind one giant cutover.

7.3 What may coexist temporarily

Temporarily allowed to coexist:
	•	legacy product surfaces consuming shims,
	•	old surface-level compatibility layers,
	•	old-to-new bridging code at upper-layer boundaries.

7.4 What may not coexist as canonical

Not allowed to coexist as canonical architecture:
	•	old universal generate boundary,
	•	old mixed runtime/semantics/orchestration ownership,
	•	old compile-centered runtime assumptions,
	•	old product-shaped runtime contracts.

7.5 Preventing old contracts from surviving too long

Temporary compatibility must have all of:
	•	explicit designation as transitional,
	•	ownership by a later cleanup phase,
	•	clear removal conditions,
	•	prohibition against becoming the “practical default” architecture.

⸻

8. Compatibility strategy

8.1 Compatibility during refactor

Compatibility is allowed as a surface-level concern only.

It may be provided through:
	•	Layer 4 shims,
	•	temporary adaptation layers,
	•	request/response translators.

8.2 When to introduce shims

Shims may be introduced only when:
	•	they unblock phase sequencing,
	•	they preserve the lower-layer contracts,
	•	they do not leak legacy assumptions into new lower layers.

8.3 When to remove shims

Shims must be removed when:
	•	the target layer has been reattached cleanly,
	•	product surfaces are operating directly on the new layer boundaries,
	•	the shim no longer protects phase sequencing.

8.4 Compatibility rule

Compatibility surfaces must not become the hidden architectural center of the refactor.

⸻

9. Risk management

9.1 Risk: recontamination of lower layers

Description

Old product-shaped or orchestration-shaped contracts may re-enter Layers 1–2 during implementation pressure.

Mitigation
	•	architecture-conformance review at each phase exit,
	•	strict temporary-shim labeling,
	•	refusal to preserve old broad generate contracts.

9.2 Risk: benchmark drift

Description

Benchmark execution may drift away from the canonical protocol during iterative work.

Mitigation
	•	Phase 2 dedicated benchmark alignment,
	•	benchmark-governed validation before broad feature restoration,
	•	benchmark-class separation maintained throughout execution.

9.3 Risk: saving too much

Description

Too much old runtime code may be retained, preserving the wrong boundaries.

Mitigation
	•	fresh runtime-core subtree replacement,
	•	explicit reuse classification from the Repo Baseline Audit,
	•	aggressive rejection of old runtime contracts as canonical.

9.4 Risk: rewriting too much

Description

Stable non-runtime code may be unnecessarily discarded.

Mitigation
	•	retain stable non-runtime substrate,
	•	rebuild only the contaminated runtime center first,
	•	reattach upper layers selectively.

9.5 Risk: upper-layer reattachment before lower-layer stability

Description

Product/engine/semantics layers may be wired back too early, forcing premature changes in the core.

Mitigation
	•	hard sequencing constraints,
	•	Layer 1 + Phase 2 validation before Layer 2,
	•	Layer 2 before Layer 3,
	•	Layer 3 before Layer 4.

9.6 Risk: compatibility shims becoming permanent

Description

Temporary bridges may become de facto architecture.

Mitigation
	•	Phase 6 cleanup as a mandatory phase,
	•	explicit shim removal conditions,
	•	no compatibility structure considered canonical.

⸻

10. Sequencing constraints

The following sequencing constraints are hard.

10.1 Layer sequencing
	•	Layer 1 implementation must precede Layer 2 implementation.
	•	Layer 2 implementation must precede Layer 3 implementation.
	•	Layer 3 implementation must precede Layer 4 reattachment.

10.2 Benchmark sequencing
	•	canonical benchmark alignment must happen after Layer 1 implementation and before substantial Layer 2 restoration.
	•	higher benchmark classes must not be used to declare success for Layer 1.

10.3 Product sequencing
	•	Layer 4 reattachment must not begin as a canonical integration step before Layers 1–3 are stable enough to consume directly.

10.4 Cleanup sequencing
	•	compatibility reduction and cleanup must follow integrated Layer 4 reattachment, not precede it.

10.5 Document governance
	•	Phase 1 is governed primarily by the Performance Core Spec.
	•	Phase 2 is governed primarily by the Benchmark Protocol.
	•	Phase 3 is governed by the General Path Spec.
	•	Phase 4 is governed by the Advanced Engines Spec.
	•	Phase 5 is governed by the Product Surfaces Spec.
	•	all phases remain subject to the Foundations document and the Repo Baseline Audit.

⸻

11. Artifact map for execution

11.1 Foundations

Governs: all phases
Usage: architectural root and invariant source.

11.2 Research Dossier

Governs: Phase 1 and Phase 2 primarily
Usage: factual substrate for MLX execution constraints, compile limits, stream semantics, and mlx-lm comparison assumptions.

11.3 Performance Core Spec

Governs: Phase 1
Usage: defines Layer 1 boundaries, ownership, and invariants.

11.4 Benchmark Protocol

Governs: Phase 2 and benchmark validation in later phases
Usage: defines workload classes, fairness rules, measurement boundaries, and benchmark pass/fail logic.

11.5 Repo Baseline Audit

Governs: Phase 0 and reuse/discard decisions across all phases
Usage: determines retained substrate and fresh runtime replacement strategy.

11.6 General Path Spec

Governs: Phase 3
Usage: defines Layer 2 responsibilities and limits.

11.7 Advanced Engines Spec

Governs: Phase 4
Usage: defines Layer 3 orchestration scope and boundaries.

11.8 Product Surfaces Spec

Governs: Phase 5
Usage: defines Layer 4 as composition root and product-surface owner.

⸻

12. Completion criteria

12.1 Architecturally complete

The refactor is architecturally complete when:
	•	Layers 1–4 exist distinctly in code,
	•	the old universal generate/decode boundary is no longer canonical,
	•	composition root responsibilities live only in Layer 4,
	•	orchestration concerns live only in Layer 3,
	•	rich single-request semantics live only in Layer 2,
	•	execution discipline and active runtime ownership live only in Layer 1.

12.2 Benchmark-validated

The refactor is benchmark-validated when:
	•	the canonical fast-path benchmark has been executed under the Benchmark Protocol,
	•	fairness versus mlx-lm is documented and reproducible,
	•	regressions are understood or eliminated according to the benchmark-governed criteria,
	•	later benchmark classes validate restored broader functionality without redefining the initial fast-path success condition.

12.3 Compatibility surfaces removable

Temporary compatibility surfaces may be removed when:
	•	Layer 4 consumes the new Layers 1–3 directly,
	•	no canonical product surface depends on old runtime contracts,
	•	shims no longer serve phase sequencing,
	•	Phase 6 cleanup criteria are satisfied.

⸻

13. Open implementation questions

Only questions that remain legitimately open are listed here.
	1.	Exact branch and carve-out mechanics for the fresh runtime subtree replacement, subject to final repo verification.
	2.	Exact temporary shim shapes needed between old product surfaces and new lower layers during Phases 3–5.
	3.	Exact benchmark matrix values and tolerances once the first stable hardware/software execution environment is locked under the Benchmark Protocol.
	4.	Exact order of reattaching Layer 3 subsystems internally, if batching/speculative/prompt-cache work cannot be restored in one pass.
	5.	Exact compatibility-surface reduction timing during Phase 6, once Layer 4 reattachment is complete.
	6.	Whether any retained non-runtime subsystem requires additional interface isolation before safe reattachment.

These are implementation questions, not architectural questions. They do not reopen the canonical decisions already frozen.

⸻