# AGENTS.md

## Purpose

This file defines the default operating contract for autonomous agents working in this repository.

It is intentionally **universal and reusable across the whole project**.  
It is not tied to a single subsystem or workstream.  
Task-specific specs, phase docs, and local plans may add constraints, but this file remains the baseline operating contract.

The expected agent behavior in this repository is:

- autonomous,
- evidence-first,
- architecture-safe,
- performance-aware,
- product-aware,
- disciplined in execution,
- strong on validation,
- and conservative about scope creep.

The goal is not just to “make changes that work”.  
The goal is to produce changes that are:

- correct,
- well-bounded,
- cleanly designed,
- high-signal,
- professionally maintainable,
- and aligned with the repository’s architecture and long-term direction.

---

## Repository mission and quality bar

This repository is not a demo, not an MVP, not a toy project, and not a speculative prototype.

It is intended to become a serious, production-ready system.

All work should be judged against that bar:
- professional quality,
- production-readiness,
- strong architectural integrity,
- high performance,
- high efficiency,
- disciplined resource usage,
- operational reliability,
- and maintainability under continued growth.

Agents must not optimize for “good enough for now” if that would create throwaway-quality foundations.
They should optimize for solutions that are credible in a real long-term production codebase.

For runtime, serving, and core-path work, the expectation is not merely to work correctly.
The expectation is to be:
- fast,
- efficient,
- well-designed,
- and competitive against strong existing baselines.

Where the project can clearly surpass the reference baseline, it should aim to do so.
Where it cannot yet surpass it, it should aim for parity without degrading architecture, clarity, or long-term maintainability.

---

## Default operating mode

### Autonomous by default
Act autonomously by default.

Do **not** stop after every small step.  
Do **not** wait for human confirmation between normal slices of work.  
Do **not** ask what to do next if the next step is inferable from:

- repository state,
- current code structure,
- docs/specs,
- prior validated results,
- existing tests,
- or the active phase/workstream direction.

Instead, operate in a loop:

1. inspect,
2. understand,
3. gather evidence,
4. decide whether implementation is justified,
5. implement one bounded slice if justified,
6. validate,
7. reassess,
8. continue until a real stop condition is reached.

### Real blocker rule
Only stop or ask for input when there is a **real blocker**.

A real blocker means something like:
- the next step requires a genuine architectural or product decision not already resolved by repo docs/specs,
- required information is missing and cannot be inferred from repository state,
- a required external dependency/service/environment cannot be accessed,
- continuing would require speculative redesign beyond approved scope,
- safety or correctness would be compromised by guessing.

Normal uncertainty is **not** a blocker.  
If uncertainty can be reduced by reading code, reading docs, running probes, inspecting artifacts, or validating assumptions, do that instead of stopping.

---

## Universal work loop

Every substantial task should follow this loop unless a task-local spec explicitly defines a stricter one.

### Phase A — Orientation
Before changing code:
- inspect the relevant repository surface,
- identify authoritative modules,
- identify relevant specs/docs/tests,
- identify the active workstream type:
  - refactor,
  - bugfix,
  - runtime/performance,
  - serving/orchestration,
  - product/UI/TUI,
  - infra/tooling,
  - architectural integration.

### Phase B — Baseline / evidence read
Before implementing:
- read the current code path,
- establish current ownership and boundaries,
- identify current behavior,
- identify current validation surface,
- identify whether a baseline, dossier, or audit-style artifact is needed.

Use lightweight evidence artifacts when useful, such as:
- current-state summary
- repo baseline readback
- research dossier
- hot-path walkthrough
- fidelity audit
- surface map
- root-cause probe
- decomposition note
- ranked target shortlist
- bounded integration note
- decision memo

Not every task needs all of them.  
Every non-trivial task needs **enough evidence** that implementation is justified.

### Phase C — Evidence gate
Before implementation, decide explicitly:

- `Patch allowed`
or
- `Continue investigation`

A patch is allowed only if:
- the target is clearly identified,
- ownership and boundaries are clear,
- the slice is narrow enough,
- expected benefit or correction is meaningful,
- validation surface is known,
- and the change does not require speculative redesign.

If that bar is not met, continue investigation instead of patching.

### Phase D — Bounded implementation
When patching:
- do one coherent bounded slice at a time,
- preserve behavior unless behavior change is part of the task,
- avoid mixing unrelated cleanup,
- keep diffs small enough to reason about,
- keep responsibilities clear.

### Phase E — Validation
Validate every meaningful slice with:
- focused tests,
- relevant regression tests,
- syntax checks,
- type checks where appropriate,
- targeted probes/benchmarks when relevant.

### Phase F — Reassessment
After validation:
- inspect the new repo state,
- decide the next best step,
- continue autonomously if no real stop condition exists.

---

## Evidence-first discipline

This repository expects **investigation before implementation** for non-trivial work.

### Investigation is mandatory when
- the bug/root cause is not isolated,
- the performance target is not yet attributable,
- architecture boundaries are involved,
- product/runtime integration is involved,
- serving behavior or concurrency is involved,
- UI structure or interaction patterns are involved,
- a subsystem is being refactored,
- or there is any risk of solving the wrong problem.

### Good-enough evidence means
Evidence should answer:
- what the current behavior is,
- where ownership currently lives,
- what the actual problem is,
- why this target is better than alternatives,
- how the change will be validated,
- and what must remain unchanged.

### Lightweight artifact preference
Prefer the smallest artifact that makes the next decision clean.
Do not create documents for their own sake.
Create only the minimum evidence needed to support disciplined execution.

---

## Production-grade decision standard

Autonomous agents must make decisions with a production-grade standard, not a prototype standard.

This means:
- do not choose temporary-looking structures unless they are explicitly intended as bounded transitional seams,
- do not introduce hacks that solve only the immediate case while weakening the long-term design,
- do not frame work as “MVP” unless the repository explicitly asks for an MVP,
- do not accept demo-quality UX, runtime behavior, or code organization as an endpoint,
- do not ship changes that are merely “functionally passing” if they are structurally poor.

When choosing between alternatives, prefer the one that is:
- more robust,
- more performant,
- more maintainable,
- more architecturally coherent,
- and more defensible as production code,
provided the scope remains bounded and the evidence supports it.

The correct target is not “quickly good enough”.
The correct target is “bounded, validated, and production-credible”.

---

## Decision confidence rule

Autonomous implementation decisions must be made only when confidence is sufficiently high.

“Sufficiently high confidence” means:
- the current behavior is understood,
- the target is well isolated,
- ownership and boundaries are clear,
- the expected effect is grounded in evidence,
- the validation surface is known,
- and the change does not depend on speculative assumptions.

If confidence is not yet high enough:
- do not patch,
- do not guess,
- do not “try something promising” prematurely,
- continue investigation until the evidence gate is genuinely met.

Low-confidence changes are considered process failures, even if they occasionally work by luck.

---

## Repository architecture discipline

Respect repository layering and ownership.

### General rule
Each file, class, and function should have a clear reason to exist.

Do not blur:
- pure presentation,
- UI mechanics,
- controller/orchestration,
- runtime logic,
- product-state ownership,
- persistence,
- transport logic,
- configuration,
- or benchmark/probe logic.

### Boundary rule
When the repository already has layering, preserve it.  
When task-specific docs define layering, follow those docs.  
Do not “simplify” by collapsing boundaries that were introduced deliberately.

### No god-object rule
Do not create or enlarge:
- controller god-objects,
- store god-objects,
- widget god-objects,
- “manager” classes that accumulate unrelated responsibilities.

### Compatibility rule
During refactors, compatibility/facade modules are acceptable if they:
- preserve behavior,
- reduce migration risk,
- and are actively shrinking the old ownership.

Long-lived duplication is not acceptable.

---

## Core engineering principles

### Required principles
- **SRP** — one dominant reason to change.
- **DRY** — remove true duplication, but do not force premature abstraction.
- **SOLID** — especially clear responsibility boundaries and sane dependency direction.

### Code quality bar
Code must be:
- correct,
- maintainable,
- performant where it matters,
- compact but readable,
- elegant in its problem-solving approach,
- and aligned with repository architecture.

### Design over cleverness
Prefer:
- clear ownership,
- clear control flow,
- explicit invariants,
- stable naming,
- strong module boundaries.

Avoid:
- clever-but-brittle shortcuts,
- hidden side effects,
- abstraction layers without real payoff,
- or micro-optimizations that damage clarity.

---

## Project-adapted Python principles

These rules are tuned for this repository’s actual work: performance-sensitive core paths, structured refactors, product surfaces, serving/orchestration, and autonomous implementation.

### Algorithmic priority
Prefer sound algorithmic design over premature micro-optimization.

Ask first:
- Is the algorithm appropriate?
- Is the data structure appropriate?
- Is the hot path actually hot?
- Is there duplicated work?
- Is ownership causing avoidable cost?

### Data structure choice
Choose deliberately:
- `list` for indexed ordered collections,
- `deque` for frequent head/tail queue behavior,
- `dict` / `set` for O(1) lookup,
- `heapq` for priority ordering,
- `bisect` for ordered lookup/insertion,
- dense/specialized containers when memory density matters.

### Allocation discipline
Reduce avoidable allocations:
- avoid unnecessary copies,
- avoid rebuilding derived state if incremental update is possible,
- prefer generators when materialization is unnecessary,
- prefer streaming over eager loading when appropriate,
- avoid temporary objects on hot paths unless required for clarity/correctness.

### Pass-count discipline
Prefer one clean pass over redundant multi-pass work when practical.  
Do not contort code to force single-pass logic if it becomes harder to maintain.

### Built-ins and stdlib first
Prefer Python built-ins and optimized standard-library facilities when they are the right fit:
- `any`, `all`, `sum`, `min`, `max`, `sorted`, `enumerate`, `zip`, `reversed`
- comprehensions and generator expressions
- `itertools`, `functools`, `collections`, `operator`, `math`, `statistics`, `heapq`, `bisect`
- native methods like `str.join`, `list.extend`, `list.sort`

Do not reimplement what the standard library already does clearly and efficiently.

### Concurrency / async
Use concurrency only when justified by the workload:
- `asyncio` for genuinely I/O-bound async flows,
- `concurrent.futures` / `multiprocessing` for isolated CPU-bound work when warranted,
- `asyncio.to_thread` for isolated blocking work in async systems when appropriate.

Do not add concurrency casually.  
Do not make systems harder to reason about for speculative gain.

### Performance honesty
Do not claim performance improvement without evidence.
Do not add complexity for hypothetical speedups.
Do not optimize cold code as if it were hot.

---

## Dependency policy

Keep external dependencies to the minimum justified set.

### Prefer
1. standard library,
2. existing repository dependencies,
3. new dependency only when the benefit is clearly justified.

### Add a dependency only if it clearly improves
- correctness,
- performance,
- stability,
- or provides non-trivial functionality that would be expensive/risky to reimplement.

When using third-party libraries:
- use them idiomatically,
- align with official documentation,
- avoid partial or accidental usage patterns.

---

## Validation policy

Every meaningful slice must be validated.

### Default validation requirements
Use:
- focused unit tests for touched behavior,
- relevant regression tests,
- syntax validation,
- type checks where relevant,
- targeted probes/benchmarks where relevant.

### Remote execution rule
When a workstream depends on the approved remote validation environment, use that environment.

In this repository, **tests/validation may be executed on the approved remote machine when that is the validated path for the active workstream**.  
Do not assume local-only validation is sufficient if the workflow for that subsystem relies on a remote execution surface.

If a workstream-specific validation path exists, follow it.

### Behavior preservation rule
When a slice is intended to preserve behavior:
- validate that explicitly,
- do not assume extractions are safe without regression coverage.

---

## Commit discipline

A slice is ready to commit only if:
- the diff is coherent,
- the slice is bounded,
- validation passes,
- unrelated artifacts are excluded,
- and the diff matches the intended scope.

### Commit message style
Use short, focused messages that describe the actual slice.

Examples:
- `refactor: extract chat presentation and completion helpers`
- `refactor: add minimal chat session store`
- `refactor: move chat shell into tui package`

---

## Workstream-specific decision sources

This file is universal.  
But some parts of the repository already have stronger local decision sources.

### Core / runtime / performance work
For core/runtime/performance decisions, the agent must explicitly consider the documents under:

- `docs/refactor/`

Especially when the task touches:
- architecture boundaries,
- runtime layering,
- benchmark semantics,
- product/runtime separation,
- serving/orchestration design,
- core execution paths,
- canonical surfaces.

For this work, the agent should not decide based only on current code shape.  
It must also use the relevant `docs/refactor/*` documents as decision inputs.

### MLXs strategic objective

For MLXs work, a central repository objective is to become the practical new standard relative to `mlx_lm`.

This means autonomous decisions should be evaluated not only on local correctness, but on whether they move the system toward:
- stronger core execution,
- stronger real-path performance,
- stronger serving/orchestration behavior,
- stronger operational reliability,
- and stronger production readiness.

The performance standard is therefore:

- where MLXs can clearly beat `mlx_lm`, it should aim to beat it;
- where it cannot yet beat `mlx_lm`, it should aim for parity;
- it should not accept unnecessary regressions in architecture, clarity, or correctness just to chase a narrow benchmark win;
- it should not settle for avoidable underperformance where a stronger design is realistically achievable.

For core/runtime/serving tasks, agents must think in terms of:
- real competitiveness,
- efficient execution,
- clean hot paths,
- scalable serving behavior,
- and production-grade system quality.

Do not optimize only for isolated micro-benchmarks.
Optimize toward making MLXs the stronger practical system.

### How to use `docs/refactor/`
For core/runtime/performance tasks:
1. identify the relevant `docs/refactor/*` docs,
2. read the current repo path against those docs,
3. use them as architecture/governance sources,
4. then perform the evidence gate.

If a current code path and `docs/refactor/*` disagree, do not “fix by instinct”.  
First determine whether:
- the docs are authoritative for that task,
- the code is lagging the intended design,
- or the repo has already legitimately moved beyond the doc.

### Core decision pattern
For core/runtime/performance work, the expected pattern is:

1. repo baseline read,
2. relevant `docs/refactor/*` read,
3. evidence artifact if needed,
4. evidence gate,
5. bounded implementation,
6. remote/local validation as appropriate,
7. reassessment.

This is especially important for GPT-5.4-class agents operating autonomously, because strong autonomy without doc-grounded decision-making can drift into solving the wrong layer or the wrong surface.

---

## Product / UI / TUI guardrails

When work touches user-facing surfaces, the product quality bar is high.

### UX / DX bar
The UI and interaction model must feel:
- modern,
- professional,
- calm,
- polished,
- visually ordered,
- readable over long sessions,
- and intentional in hierarchy and spacing.

### Must not drift into
- demo UI,
- MVP roughness,
- dashboard-wall clutter,
- terminal nostalgia aesthetics,
- sysadmin-tool visual noise,
- fake complexity,
- transcript spam for things that belong in dedicated surfaces.

### UI principles
- persistent UI should be restrained,
- primary content should remain dominant,
- secondary information should remain secondary,
- spacing and grouping should be deliberate,
- overlays/pickers/palette should feel like product UI, not hacks,
- status/progress/help/errors should be honest and calm.

### Honesty rule
Do not fake:
- progress,
- steps,
- tools,
- diagnostics,
- or runtime state
unless they are genuinely backed by the system.

---

## Refactor discipline

Refactors should be:
- phased,
- bounded,
- behavior-preserving by default,
- and test-supported.

### Preferred shape
- extract pure logic first,
- shrink oversized modules gradually,
- introduce small compatibility seams when needed,
- validate at every step,
- move one responsibility at a time.

### Do not
- mix multiple future phases into one patch,
- combine refactor and UX redesign unless explicitly authorized,
- combine refactor and persistence expansion unless explicitly authorized,
- bundle unrelated cleanup just because a file is open.

---

## End-of-slice decision rule

After each completed slice/session:
1. summarize current state,
2. decide the next best slice/session,
3. continue immediately if no real stop condition exists.

The next step should be chosen based on:
- evidence readiness,
- architectural leverage,
- clean ownership,
- correctness,
- and the active phase/workstream direction.

Do not stop simply because one slice finished.

---

## Real stop conditions

These are the only valid stop conditions:

1. A real blocker cannot be resolved from repository state or available tools.
2. The next step requires a genuine architectural or product decision not already resolved by docs/specs.
3. A phase boundary has been reached and the next phase needs an explicit direction choice.
4. Further autonomous slicing would be low-value without review.

Everything else is **not** a stop condition.

---

## GPT-5.4-class autonomy expectations

This repository expects strong autonomous agents to be capable of:
- reading the repo before acting,
- consulting authoritative docs before deciding,
- creating the minimum sufficient evidence,
- making bounded decisions,
- implementing narrow slices,
- validating thoroughly,
- continuing independently,
- and stopping only for real reasons.

The expected default posture is:
- autonomous,
- evidence-first,
- disciplined,
- architecture-safe,
- honest,
- and relentless about clean forward progress.

If the next step is obvious, do it.  
If the next step is not obvious, investigate.  
If a real stop condition is not present, do not stop.