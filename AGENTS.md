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

The goal is not just to "make changes that work".  
The goal is to produce changes that are:

- correct,
- well-bounded,
- cleanly designed,
- high-signal,
- professionally maintainable,
- and aligned with the repository's architecture and long-term direction.

---

## Project identity

This repository is a **next-generation MLX inference and serving system**.

Current active qualification:
- an **accepted MLXs baseline with the active performance-architecture redesign running under `docs/execution/`**
- the current codebase is a trustworthy baseline, not the long-term architecture or performance endpoint
- live current-state authority is `docs/execution/STATUS.md`

It is:
- an **MLX inference and serving** system
- performance-first
- production-first
- explicit-state and explicit-side-effects
- strongly layered
- replayable and recoverable by design
- performance-conscious by architecture
- observable and debuggable
- contract-first across load / generate / cache / batch / server
- benchmark-driven and validation-heavy
- built for low architectural regret

It is **not**:
- a research-only benchmark sandbox
- a training or fine-tuning repository
- a thin `mlx_lm` wrapper
- a speculative architecture playground
- a generic multi-backend runtime by default
- a browser-first UI project
- a place to import architecture from previous projects unless justified by this repo

## Current project status

Official status:

**CURRENT IMPLEMENTED BASELINE: ACCEPTED CONTROL UNDER ACTIVE REDESIGN**

Interpretation:
- the current implemented baseline is the accepted control and is production-freezeable
- the architecture direction is governed by the live project docs under `docs/execution/` and `docs/refactor/`
- the repository currently operates under the active milestone and workstream recorded in `docs/execution/STATUS.md`
- allowed work from now on is limited to:
  - active milestone execution
  - approved architecture-definition work
  - real bugfixes
  - reliability hardening
  - performance hardening
  - operational completion of already-modeled paths
  - evidence-driven improvements
- if a change would materially alter core architecture outside the active approved direction, stop and escalate instead of improvising

## Repository scope policy

This repository must converge to and remain a **single-purpose MLX inference and serving repository**.

Rules:
- no archive or historical surface should act as live architecture authority
- no backward-compatibility scaffolding inside core paths unless it serves an active documented requirement
- no historical preservation inside live core paths once removability is verified
- no non-MLXs surfaces unless they are concretely required by active implementation, tests, benchmark harnesses, tooling, or operator guidance
- no parallel authority trees
- no speculative subsystem expansion without a file-backed reason
- no preserving obsolete code "just in case" once removability is verified

Cleanup rule:
- define the canonical keep-list first
- verify removability via imports, tests, benchmarks, tooling, frozen docs, and current workflow
- once a surface is verified as non-required, remove it instead of preserving dead weight

## Binding source order

When ambiguity exists, use this precedence order:

1. `docs/execution/STATUS.md`
2. active milestone files under `docs/execution/milestones/`
3. `docs/execution/REDESIGN_MODE.md`
4. `docs/execution/REFACTOR_POLICY.md`
5. `docs/execution/FREEZE.md`
6. `docs/execution/DECISIONS.md`
7. `docs/execution/WORKLOG.md`
8. `docs/specs.md`
9. relevant canonical documents under `docs/refactor/`
10. `docs/execution/CODEX_GUIDE.md` and the playbooks under `docs/execution/playbooks/`
11. `docs/execution/ROADMAP.md`
12. `docs/execution/BACKLOG.md`
13. `docs/execution/REPO_MAP.md` and `docs/execution/INDEX.md`
14. code reality of this repository

Never override a higher-priority source with local convenience.

## Project control surfaces

The following files are the live control surfaces for ongoing work:

- `docs/execution/STATUS.md` = active milestone, workstream, blockers, and immediate next step
- active milestone files under `docs/execution/milestones/` = execution truth for the active front, including `RESEARCH.md`, `PLAN.md`, `TASKS.md`, `RISKS.md`, `REPORT.md`, and supporting artifacts
- `docs/execution/REDESIGN_MODE.md` = binding redesign governance and stop/continue law
- `docs/execution/REFACTOR_POLICY.md` = binding front-selection, rerank, and continuation policy
- `docs/execution/FREEZE.md` = frozen seams, rejected shapes, reopen rules, and accepted baselines
- `docs/execution/DECISIONS.md` = durable accepted/rejected choices
- `docs/execution/WORKLOG.md` = validated slice history, artifact paths, worktrees, and useful execution history
- `docs/specs.md` = library-level product/specification source of truth
- relevant documents under `docs/refactor/` = canonical architecture/runtime/performance inputs
- `docs/execution/ROADMAP.md` = milestone order and exit framing
- `docs/execution/BACKLOG.md` = queued work not yet active
- `docs/execution/CODEX_GUIDE.md` = local Codex operating guidance
- `docs/execution/playbooks/file-backed-ops.md` = mandatory file-backed update policy
- `docs/execution/playbooks/remote-validation.md` = authoritative remote validation workflow
- `docs/execution/playbooks/worktree-policy.md` = worktree isolation policy
- `docs/execution/playbooks/benchmarking.md` = benchmark workflow
- `docs/execution/REPO_MAP.md` and `docs/execution/INDEX.md` = repo navigation aids

These files are part of the active operating surface of the repository.

Do not let them drift from code reality.  
Do not create parallel planning surfaces unless explicitly authorized.

## File-backed execution system

`docs/execution/` is the file-backed operating system of record.

Rules:
- no critical state only in chat
- after meaningful progress, update the right execution file immediately
- keep files dense, operational, and link-heavy
- `STATUS.md` is the summary, not the full history
- `WORKLOG.md` holds slice history and artifact paths
- `FREEZE.md` changes only when a family materially closes, blocks, or reopens
- if file-backed truth and chat memory disagree, file-backed truth wins

Update order after meaningful progress:
1. update `STATUS.md` when active state changes
2. update the active milestone `TASKS.md` and `RISKS.md` when work advances
3. append durable choices to `DECISIONS.md`
4. append slice history and artifact paths to `WORKLOG.md`
5. update `FREEZE.md` only when a family materially closes, blocks, or reopens

## Benchmark and remote validation discipline

For performance, serving, or promotable runtime claims:
- local is for reading, editing, focused tests, smoke checks, and local probes
- remote is authoritative for promotable performance or live-behavior claims when the workstream uses that path
- serialize heavy remote benchmarks and avoid conflicting remote slices
- preserve the accepted remote baseline and use isolated remote candidates for risky work
- keep benchmark and probe artifact paths in `docs/execution/WORKLOG.md`

## Repository mission and quality bar

This repository is not a demo, not an MVP, not a toy project, and not a speculative prototype.

It is intended to become a serious, production-ready system.

At the product/spec level, the target is to outperform `mlx_lm` on Apple Silicon while remaining configurable, feature-complete where relevant, and architecturally disciplined.

All work should be judged against that bar:
- professional quality
- production-readiness
- strong architectural integrity
- high performance
- high efficiency
- disciplined resource usage
- operational reliability
- and maintainability under continued growth

Agents must not optimize for "good enough for now" if that would create throwaway-quality foundations.
They should optimize for solutions that are credible in a real long-term production codebase.

For runtime, serving, and core-path work, the expectation is not merely to work correctly.
The expectation is to be:
- fast
- efficient
- well-designed
- and competitive against strong existing baselines

Where the project can clearly surpass the reference baseline, it should aim to do so.
Where it cannot yet surpass it, it should aim for parity without degrading architecture, clarity, or long-term maintainability.

## Default operating mode

### MLXs redesign override
For MLXs performance redesign work, follow these as binding governance sources:
- `docs/execution/REDESIGN_MODE.md`
- `docs/execution/REFACTOR_POLICY.md`

They override the default slice-by-slice posture for MLXs runtime/performance sequencing, front selection, and stop conditions.

### Autonomous by default
Act autonomously by default.

Do **not** stop after every small step.  
Do **not** wait for human confirmation between normal slices of work.  
Do **not** ask what to do next if the next step is inferable from:

- repository state
- current code structure
- docs/specs
- prior validated results
- existing tests
- or the active phase/workstream direction

Instead, operate in a loop:

1. inspect
2. understand
3. gather evidence
4. decide whether implementation is justified
5. implement one bounded slice if justified
6. validate
7. reassess
8. continue until a real stop condition is reached

### MLXs front-management rule
For MLXs runtime/performance/refactor work:

- Small-step work is allowed only to locate and confirm a bottleneck.
- Mini-family or mini-refactor chaining is not the default behavior.
- If a front is already well understood and has burned roughly 2-3 bounded families without a strong promotable win, the default next move must be exactly one of:
  - an integrated refactor of the relevant layer or contract
  - honest closure of that front and a global rerank
- Do not generate long `Rxx-S1/S2/S3/...` chains on the same local seam unless the file-backed system explicitly records why the front is still only in bottleneck-confirmation mode.
- Keep one main runtime/refactor front active at a time.
- A secondary research or instrumentation front is allowed only when it directly supports the active main front.
- Codex chooses the next path autonomously from file-backed truth.
- Once the direction is established, the user does not define slices by default.
- On success: promote or commit the winning state if appropriate, update file-backed docs, and continue immediately.
- On failure: reject or revert cleanly, update file-backed docs, and continue immediately.

### MLXs worktree / branch hygiene rule
For MLXs runtime/performance/refactor work:

- Every risky redesign path must use a dedicated worktree and branch.
- Each worktree must map to one active path only.
- When a new path opens, choose whether to reuse the current technical worktree or open a fresh one, and record that decision in the file-backed execution system.
- When a path wins:
  - promote or commit the winning state correctly
  - update the file-backed docs
  - remove obsolete worktrees and branches by default once they are no longer needed
- When a path loses:
  - reject or revert cleanly
  - update the file-backed docs
  - remove the closed worktree and branch by default unless there is a specific recorded reason to retain them
- Do not accumulate stale worktrees or stale branches.
- If a worktree or branch is intentionally retained after closure, record the exact reason in the file-backed execution docs.

### Real blocker rule
Only stop or ask for input when there is a **real blocker**.

A real blocker means something like:
- the next step requires a genuine architectural or product decision not already resolved by repo docs/specs
- required information is missing and cannot be inferred from repository state
- a required external dependency/service/environment cannot be accessed
- continuing would require speculative redesign beyond approved scope
- safety or correctness would be compromised by guessing

Normal uncertainty is **not** a blocker.  
If uncertainty can be reduced by reading code, reading docs, running probes, inspecting artifacts, or validating assumptions, do that instead of stopping.

## Universal work loop

Every substantial task should follow this loop unless a task-local spec explicitly defines a stricter one.

### Phase A - Orientation
Before changing code:
- inspect the relevant repository surface
- identify authoritative modules
- identify relevant specs/docs/tests
- identify the active workstream type:
  - refactor
  - bugfix
  - runtime/performance
  - serving/orchestration
  - product/UI/TUI
  - infra/tooling
  - architectural integration

### Phase B - Baseline / evidence read
Before implementing:
- read the current code path
- establish current ownership and boundaries
- identify current behavior
- identify current validation surface
- identify whether a baseline, dossier, or audit-style artifact is needed

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

### Phase C - Evidence gate
Before implementation, decide explicitly:

- `Patch allowed`
or
- `Continue investigation`

A patch is allowed only if:
- the target is clearly identified
- ownership and boundaries are clear
- the slice is narrow enough
- expected benefit or correction is meaningful
- validation surface is known
- and the change does not require speculative redesign

If that bar is not met, continue investigation instead of patching.

### Phase D - Bounded implementation
When patching:
- do one coherent bounded slice at a time
- preserve behavior unless behavior change is part of the task
- avoid mixing unrelated cleanup
- keep diffs small enough to reason about
- keep responsibilities clear

### Phase E - Validation
Validate every meaningful slice with:
- focused tests
- relevant regression tests
- syntax checks
- type checks where appropriate
- targeted probes/benchmarks when relevant

### Phase F - Reassessment
After validation:
- inspect the new repo state
- decide the next best step
- continue autonomously if no real stop condition exists

For MLXs runtime/performance work, reassessment must also answer:
- is this front still in bottleneck-confirmation mode?
- or is it already sufficiently understood that the next move must be integrated refactor or closure/rerank?

Do not answer that question by inertia. Answer it from the file-backed evidence.

## Evidence-first discipline

This repository expects **investigation before implementation** for non-trivial work.

### Investigation is mandatory when
- the bug/root cause is not isolated
- the performance target is not yet attributable
- architecture boundaries are involved
- product/runtime integration is involved
- serving behavior or concurrency is involved
- UI structure or interaction patterns are involved
- a subsystem is being refactored
- or there is any risk of solving the wrong problem

### Good-enough evidence means
Evidence should answer:
- what the current behavior is
- where ownership currently lives
- what the actual problem is
- why this target is better than alternatives
- how the change will be validated
- and what must remain unchanged

### Lightweight artifact preference
Prefer the smallest artifact that makes the next decision clean.
Do not create documents for their own sake.
Create only the minimum evidence needed to support disciplined execution.

## Production-grade decision standard

Autonomous agents must make decisions with a production-grade standard, not a prototype standard.

This means:
- do not choose temporary-looking structures unless they are explicitly intended as bounded transitional seams
- do not introduce hacks that solve only the immediate case while weakening the long-term design
- do not frame work as "MVP" unless the repository explicitly asks for an MVP
- do not accept demo-quality UX, runtime behavior, or code organization as an endpoint
- do not ship changes that are merely "functionally passing" if they are structurally poor

When choosing between alternatives, prefer the one that is:
- more robust
- more performant
- more maintainable
- more architecturally coherent
- and more defensible as production code,
provided the scope remains bounded and the evidence supports it.

The correct target is not "quickly good enough".
The correct target is "bounded, validated, and production-credible".

## Decision confidence rule

Autonomous implementation decisions must be made only when confidence is sufficiently high.

"Sufficiently high confidence" means:
- the current behavior is understood
- the target is well isolated
- ownership and boundaries are clear
- the expected effect is grounded in evidence
- the validation surface is known
- and the change does not depend on speculative assumptions

If confidence is not yet high enough:
- do not patch
- do not guess
- do not "try something promising" prematurely
- continue investigation until the evidence gate is genuinely met

Low-confidence changes are considered process failures, even if they occasionally work by luck.

## Core framing

The intended model is:

- the **accepted baseline** is the control, not the architecture endpoint
- the **MLXs runtime core** is the product center
- the **load / generate / cache / batch / server** boundaries are real and must stay explicit
- the active redesign is governed through the file-backed execution system under `docs/execution/`
- `docs/refactor/` provides architecture and performance governance inputs for hot-path work
- benchmark harnesses and probes measure surfaces; they do not define product semantics or canonical state
- the local accepted baseline and risky redesign candidates must remain separated
- `mlx_lm` is the comparison baseline, not the architecture authority
- canonical truth lives in MLXs code, explicit contracts, and the live execution docs
- raw benchmark, probe, remote-shell, or ad hoc measurement payloads must never become canonical state
- model-family integrations belong behind explicit contracts and must not widen the hot path without evidence
- server / CLI / chat / operator surfaces matter, but they must not own runtime truth
- remote-authoritative validation is first-class for promotable performance claims

## Mandatory engineering principles

These are always binding:

- **SOLID**
- **SRP**
- **DRY**
- explicit state over implicit state
- explicit side effects over hidden behavior
- typed boundaries over stringly-typed protocols
- fail-closed behavior on critical paths
- replayability and recovery as first-class requirements
- architectural optimization before micro-optimization
- minimal external dependencies
- evidence before redesign
- measurable behavior over performance folklore

### Simplicity rule

Do **not** optimize for shallow simplicity if it weakens stronger constraints.

Use the simplest solution only when it also preserves:
- correctness
- reliability
- strong boundaries
- replayability
- recovery guarantees
- performance requirements
- maintainability

No simplistic solutions. No complexity theater either.

## Structural expectations

This project must not become flat or convenience-driven.

The codebase should be:
- deeply structured where responsibilities require it
- compact where complexity is not justified
- explicit in ownership per file, class, and function
- decomposed by responsibility, heat, and side-effect boundaries
- sophisticated in design where necessary
- conservative in abstraction where evidence is weak

Do not collapse boundaries to reduce file count.  
Do not add layers purely for style.  
Do not create controller god-objects, store god-objects, widget god-objects, or "manager" classes that accumulate unrelated responsibilities.  
Compatibility/facade modules are acceptable only if they preserve behavior, reduce migration risk, and are actively shrinking the old ownership. Long-lived duplication is not acceptable.

Core components should depend on narrow contracts/interfaces/protocols rather than concrete implementations.

This is especially important across:
- model loading
- generation
- cache
- batching
- server / transport
- architecture-specific model integrations

Adding a new model family, cache backend, or operator surface should not require tangling the core hot path or widening low-level contracts unnecessarily.

## Non-negotiable rules

### Never do these
- do not redesign outside the approved architecture direction
- do not import previous-project architecture without concrete repo evidence
- do not move logic across layers for convenience
- do not introduce hidden side effects
- do not confuse the accepted control baseline with the target redesign direction
- do not let stale benchmark notes, stale reranks, or superseded docs steer current work
- do not use raw benchmark, probe, remote-shell, or ad hoc measurement payloads as canonical state
- do not add speculative abstractions
- do not broaden scope without evidence
- do not archive stale material instead of deleting it
- do not keep backward-compatibility shims for removed non-MLXs surfaces
- do not preserve stale docs "for reference" in the active repository
- do not maintain parallel documentation or authority trees
- do not keep non-MLXs surfaces without concrete evidence that the current MLXs baseline still requires them
- do not mutate protected branches directly; work from a dedicated branch and, when the redesign policy requires it, a dedicated worktree
- do not use extra git worktrees casually or decoratively; use them when isolation improves safety or when the redesign policy requires them
- do not bypass the file-backed execution system, remote validation gates, accepted-baseline isolation, or promotion/freeze contracts
- do not use `shell=True`
- do not omit subprocess timeouts
- do not use try/except to hide required setup or structural failures
- do not fake cleanup/refactors that drift from the repository purpose

### When uncertain
- inspect the repository first
- inspect the binding documents
- derive the keep-list from live MLXs code, active tests, benchmark harnesses, tooling, and operator guidance
- prefer the smallest correct repository-native step
- if confidence is below 90%, investigate instead of guessing
- if no acceptable solution satisfies the active constraints, stop and surface the conflict explicitly

## Performance and optimization rules

Performance is not a polishing phase. It is an architectural property.

### General rules
- optimize architecture before micro-optimizing code
- prefer bounded work over heuristics
- prefer exact filters before semantic/model-backed work
- prefer reuse over recomputation
- prefer batching over chatty I/O
- prefer references/indirection over copying large payloads
- prefer predictable latency over flashy best-case throughput
- disabled instrumentation should have near-zero overhead

### Mandatory Python principles

- Enforce **SOLID**, **DRY**, and **SRP** strictly.
- Code must be:
  - correct, robust, and maintainable
  - highly performant and well optimized for time and memory
  - compact yet readable
  - clearly separated in responsibility per file, class, and function
  - **elegant** in solution choice and implementation

#### Elegance (mandatory)

- **Elegance** means fully solving the problem with the least *accidental* complexity while preserving correctness, clarity, robustness, performance, and extensibility.
- When outcomes are equivalent, prefer the solution that is:
  - easier to understand and justify
  - cleaner architecturally
  - composed of fewer components, abstraction layers, states, branches, and unnecessary special cases
  - better separated in responsibilities
  - lower in coupling and smaller in API surface
  - more natural to maintain, test, and evolve
- Elegance does **not** mean removing required features, code golf, artificial compression, or avoiding useful abstractions.
- Avoid overengineering, premature abstraction, unnecessary patterns, and solutions disproportionate to the problem.
- Always seek the correct leverage point: prefer mechanisms, proxies, models, or abstractions that achieve the result in the most direct, clean, and efficient way.
- When greater complexity is truly required for correctness, reliability, performance, scalability, or real architectural constraints, that complexity is allowed only if it is clearly justified, localized, and contained.

#### Performance and optimization

- Prefer sound algorithmic design (Big-O) over premature micro-optimization.
- Always consider time complexity, space complexity, data access patterns, allocation volume, and total I/O cost.
- Choose appropriate data structures:
  - `list` for small/moderate sequences with index access
  - `deque` for queues/stacks with frequent head operations
  - `dict` and `set` for frequent O(1) lookups
  - `heapq` for priority queues
  - `bisect` for searches over sorted sequences
  - `array` / specialized types when memory density matters
  - extend the same discipline to other structures when they clearly fit the access pattern
- Avoid redundant work:
  - do not recompute results already available (local memoization/caching when justified)
  - avoid unnecessary copies/conversions of lists, strings, or structures
  - prefer a single pass over multiple passes when equivalent
- Reduce allocations:
  - use generator expressions when the full materialized list is not needed
  - stream inputs (files, network, large datasets) when practical
- Optimize I/O:
  - batch operations instead of repeated single calls
  - use adequate read/write buffering
- Use logical short-circuiting (`and`, `or`, `any`, `all`) to skip unnecessary computation.
- When appropriate, consider:
  - `concurrent.futures` or `multiprocessing` for isolatable CPU-bound work
  - `asyncio` for I/O-bound workloads with many waits
  - `asyncio.to_thread` to avoid blocking the loop with light blocking calls

#### Real execution and hardware efficiency

- Maximize real-world efficiency within the Python interpreter and platform limits.
- Reduce interpreter overhead, unnecessary allocations and copies, redundant cross-layer hops, and pure-Python reimplementation when builtins, stdlib, or native-backed libraries are faster and clearer.
- Prefer approaches that improve data locality, batching, streaming, appropriate parallelism, and efficient use of CPU, memory, cache, and system I/O.
- Avoid solutions that are correct but introduce structural overhead disproportionate to the value delivered.

#### Operational predictability and stability under load

- Prefer solutions with stable, predictable performance under realistic load, not only fast on small inputs or average cases.
- Avoid hidden complexity, unbounded memory growth, uncontrolled buffers or caches, and sharp degradations in throughput, latency, or tail latency.
- Always consider behavior on large datasets, hot paths, concurrent workloads, and realistic scenarios.

#### Builtins and C-backed stdlib

- Prefer builtins where they fit: `map`, `any`, `all`, `filter`, `sum`, `min`, `max`, `sorted`, `enumerate`, `zip`, `reversed`, and similar.
- Prefer C-optimized stdlib: `itertools`, `functools`, `collections`, `operator`, `math`, `statistics`, `heapq`, `bisect`.
- Use list/dict/set comprehensions and generator expressions for compact, fast code when readable.
- Prefer native methods (`str.join`, `list.extend`, `list.sort`, etc.) over reimplementing the same logic in pure Python.
- Use other stdlib primitives that reduce overhead and accidental complexity when they clearly apply.

#### External dependencies

- Keep third-party dependencies to the minimum needed.
- Prefer C/Rust-optimized libraries over slow pure-Python equivalents when the tradeoff is clear.
- Add external libraries only for clear benefit (performance, stability, non-trivial functionality).
- Use external APIs idiomatically, aligned with official documentation.

#### Final choice rule

Among correct solutions compatible with requirements, prefer the one that maximizes structural clarity, separation of responsibilities, architectural simplicity, and efficiency while minimizing accidental complexity.

### I/O rules
- batch operations where possible
- use buffering appropriately
- stream large inputs when practical
- minimize syscall chatter on hot operational paths

### Hardware / OS awareness
Be hardware- and OS-aware when the benefit is real and measurable.

Examples:
- process spawn cost
- filesystem layout and path locality
- buffering and file access patterns
- syscall count
- thread/process overhead
- concurrency model selection
- memory layout and container overhead
- use of C-backed stdlib functionality

Do not introduce platform-sensitive tricks without evidence and containment.

### Optimization discipline
Advanced optimizations are welcome only when they are:
- measurable
- localized
- explainable
- maintainable
- compatible with replay, debugging, and recovery

Benchmark or remove.

## Layer boundaries

### `src/mlxs/runtime_core/`
- hot-path runtime state, contracts, selection, greedy/prefill/decode, and batch progression only
- no product-surface orchestration
- no benchmark-only logic
- no server transport or chat UI concerns

### `src/mlxs/generate/`
- single-request generation helpers, compile hooks, logits/sampling/stop helpers only
- no server / operator surface semantics
- no benchmark authority

### `src/mlxs/batch/`
- batch scheduler, active-batch progression, and stream coordination only
- no CLI / API semantics
- no benchmark-specific branching as product logic

### `src/mlxs/cache/` and `src/mlxs/prompt_cache/`
- cache residency, masks, quantization, eviction, and memory behavior only
- no operator surface logic
- no benchmark-only ownership decisions

### `src/mlxs/load/`
- model/tokenizer/format/weight resolution and loading only
- no serving orchestration
- no runtime policy unrelated to loading

### `src/mlxs/models/` and `src/mlxs/layers/`
- architecture-specific model implementations and reusable neural-network layers only
- no benchmark harness semantics
- no product-surface branching
- no operator/UI concerns

### `src/mlxs/protocols/`
- narrow contracts and interfaces only
- no concrete orchestration
- no hot-path logic

### `src/mlxs/general_path/`
- shared single-request flow-control seams only
- no server transport
- no benchmark authority

### `src/mlxs/advanced_engines/` and `src/mlxs/speculative/`
- optional advanced execution features only
- no baseline contract drift without explicit docs, tests, and evidence

### `src/mlxs/product_surfaces/`
- API / CLI / operator surface composition only
- may call lower layers
- must not reimplement runtime core, cache, or batch semantics

### `src/mlxs/server/`
- HTTP transport, queueing, SSE, dependency wiring, and boundary handling only
- no core runtime ownership
- no model-family business logic

### `src/mlxs/chat/` and `src/mlxs/ui/`
- chat/UI presentation, interaction, and session scaffolding only
- no canonical runtime ownership
- no benchmark or hot-path policy

### `src/mlxs/convert/`
- model-format conversion planning, execution, diagnostics, and verification only
- no inference hot-path semantics
- no serving/product ownership

### `src/mlxs/config/`
- schema, loader, and CLI config handling only
- no hidden runtime side effects

### `src/mlxs/observability/`
- metrics and logging surfaces only
- measure, do not judge or orchestrate

### `benchmarks/mlxs_vs_mlx_lm/`
- measurement, comparison, and probe harnesses only
- no canonical runtime authority
- no product-semantics branching

## Freeze discipline

Work must proceed in steps.

Each step must:
1. define exact scope
2. define boundaries
3. implement only authorized work
4. run relevant tests and checks
5. undergo anti-drift review
6. be explicitly accepted / frozen before the next step becomes active
7. record material family closure/blockage in `docs/execution/FREEZE.md` when applicable

Do not reopen frozen steps without concrete evidence.

If a frozen step has a real issue:
- patch locally
- prove the fix
- re-freeze
- do not reopen broad design unless necessary

## Phase activation rule

Do not start a new phase or milestone just because it appears next in the roadmap.

A phase becomes active only when:
- it is the current approved milestone/workstream in `docs/execution/STATUS.md`
- it is compatible with `docs/execution/ROADMAP.md`
- its work is reflected in the active milestone files, especially `TASKS.md` and `RISKS.md`
- no unresolved blocker from the previous frozen step remains

Do not silently activate future phases.  
Do not blend multiple phases or milestones into one execution pass unless explicitly authorized.

## Testing expectations

Before claiming work complete, run the smallest relevant real checks.

Typical commands:

### focused runtime / batch
`uv run pytest -q tests/unit/test_runtime_core tests/unit/test_batch tests/unit/test_generate`

### server / product surfaces
`uv run pytest -q tests/unit/test_server tests/unit/test_product_surfaces tests/integration/test_cli_e2e.py`

### models / load / cache / convert
`uv run pytest -q tests/unit/test_models tests/unit/test_load tests/unit/test_cache tests/unit/test_convert`

### benchmark tooling
`uv run pytest -q tests/unit/test_benchmarks`

### type check
`uv run mypy src/mlxs`

### lint
`uv run ruff check src/mlxs tests benchmarks/mlxs_vs_mlx_lm`

### compile smoke
`python3 -m compileall src/mlxs tests benchmarks/mlxs_vs_mlx_lm`

Green tests are necessary, not sufficient.

## Definition of done

A task is done only if:
- it stays within the authorized scope
- layer boundaries remain intact
- no architectural drift is introduced
- relevant tests pass
- type-check and lint pass where applicable
- no hidden side effects were introduced
- no future phase or family was silently pre-implemented
- the result is consistent with the active control files
- remote-authoritative validation was used when the workstream requires it
- performance and reliability constraints are not weakened

## Commit discipline

A slice is ready to commit only if:
- the diff is coherent
- the slice is bounded
- validation passes
- unrelated artifacts are excluded
- and the diff matches the intended scope

### Commit message style
Use short, focused messages that describe the actual slice.

Examples:
- `refactor: extract layer-1 progression contract`
- `perf: reduce batch-step allocation churn`
- `fix: preserve exact parity in staggered admission path`

## Roadmap labeling rule

Whenever proposing next steps, explicitly label them as one of:
- **local milestone roadmap**
- **project-level roadmap**

If scope drift is present, flag it explicitly.

## Strategic context usage

The repository may use broader runtime-quality principles as strategic context, but this repo is still a single-purpose MLX inference and serving system.

Do not silently expand the project into:
- a general-purpose ML runtime
- a multi-backend platform
- a training or fine-tuning platform
- a general-purpose agent platform
- an enterprise platform

unless that expansion is explicitly authorized.

## Workstream-specific decision sources

This file is universal.  
But some parts of the repository already have stronger local decision sources.

### Core / runtime / performance work
For core/runtime/performance decisions, the agent must explicitly consider the documents under:

- `docs/refactor/`

Especially when the task touches:
- architecture boundaries
- runtime layering
- benchmark semantics
- product/runtime separation
- serving/orchestration design
- core execution paths
- canonical surfaces

For this work, the agent should not decide based only on current code shape.  
It must also use the relevant `docs/refactor/*` documents as decision inputs.

### MLXs strategic objective

For MLXs work, a central repository objective is to become the practical new standard relative to `mlx_lm`.

This means autonomous decisions should be evaluated not only on local correctness, but on whether they move the system toward:
- stronger core execution
- stronger real-path performance
- stronger serving/orchestration behavior
- stronger operational reliability
- and stronger production readiness

The performance standard is therefore:

- where MLXs can clearly beat `mlx_lm`, it should aim to beat it
- where it cannot yet beat `mlx_lm`, it should aim for parity
- it should not accept unnecessary regressions in architecture, clarity, or correctness just to chase a narrow benchmark win
- it should not settle for avoidable underperformance where a stronger design is realistically achievable

For core/runtime/serving tasks, agents must think in terms of:
- real competitiveness
- efficient execution
- clean hot paths
- scalable serving behavior
- and production-grade system quality

Do not optimize only for isolated micro-benchmarks.
Optimize toward making MLXs the stronger practical system.

### How to use `docs/refactor/`
For core/runtime/performance tasks:
1. identify the relevant `docs/refactor/*` docs
2. read the current repo path against those docs
3. use them as architecture/governance sources
4. then perform the evidence gate

If a current code path and `docs/refactor/*` disagree, do not "fix by instinct".  
First determine whether:
- the docs are authoritative for that task
- the code is lagging the intended design
- or the repo has already legitimately moved beyond the doc

### Core decision pattern
For core/runtime/performance work, the expected pattern is:

1. repo baseline read
2. relevant `docs/refactor/*` read
3. evidence artifact if needed
4. evidence gate
5. bounded implementation
6. remote/local validation as appropriate
7. reassessment

This is especially important for GPT-5.4-class agents operating autonomously, because strong autonomy without doc-grounded decision-making can drift into solving the wrong layer or the wrong surface.

## Product / API / CLI / operator surface guardrails

When work touches user-facing or operator-facing surfaces, the product quality bar is high.

### UX / DX bar
The API, CLI/TUI, and operator interaction model must feel:
- modern
- professional
- calm
- polished
- visually ordered where visual surfaces exist
- readable over long sessions
- and intentional in hierarchy and spacing

### Must not drift into
- demo UI or demo CLI
- MVP roughness
- dashboard-wall clutter
- terminal nostalgia aesthetics
- sysadmin-tool visual noise
- fake complexity
- log/transcript spam for things that belong in dedicated surfaces

### Surface principles
- persistent surfaces should be restrained
- primary content should remain dominant
- secondary information should remain secondary
- spacing and grouping should be deliberate
- interactive pickers/palettes/menus should feel like product surfaces, not hacks
- status/progress/help/errors should be honest and calm

### Honesty rule
Do not fake:
- progress
- steps
- tools
- diagnostics
- or runtime state
unless they are genuinely backed by the system.

## Refactor discipline

Refactors should be:
- phased
- bounded
- behavior-preserving by default
- and test-supported

### Preferred shape
- extract pure logic first
- shrink oversized modules gradually
- introduce small compatibility seams when needed
- validate at every step
- move one responsibility at a time

### Do not
- mix multiple future phases into one patch
- combine refactor and UX redesign unless explicitly authorized
- combine refactor and persistence expansion unless explicitly authorized
- bundle unrelated cleanup just because a file is open

## End-of-slice decision rule

After each completed slice/session:
1. summarize current state
2. decide the next best slice/session
3. continue immediately if no real stop condition exists

For MLXs runtime/performance work:
- if the result wins strongly enough, promote or commit it appropriately and continue immediately
- if the result loses, reject or revert it cleanly and continue immediately
- do not stop at ordinary checkpoints, reranks, family closure, or because one attempt finished
- if the front is already understood enough, do not default to another bounded family on the same seam

The next step should be chosen based on:
- evidence readiness
- architectural leverage
- clean ownership
- correctness
- and the active phase/workstream direction

Do not stop simply because one slice finished.

## Real stop conditions

These are the only valid stop conditions:

1. A real architecture-boundary blocker cannot be resolved from repository state or available tools.
2. The next step requires a genuine strategic or product-policy decision not already resolved by docs/specs.
3. A hard execution blocker prevents safe continuation.
4. Adjacent high-leverage paths are honestly exhausted.
5. Remote-authoritative validation is impossible exactly when promotion requires it.

Everything else is **not** a stop condition.

## Local decision autonomy rule

Autonomous execution is required for **local, bounded, repository-native forks**.

Do not stop for human input when the choice is between multiple next steps that are all:
- compatible with the approved architecture direction and accepted implemented baseline
- compatible with the active repository rules
- bounded in scope
- reversible if rejected
- decidable from repository state, validated work, and current evidence

Examples:
- choosing the next bounded hardening slice
- ordering nearby cleanup or refactor steps
- choosing which already-modeled lifecycle path to wire first
- sequencing closely related validation work
- deciding which small operational gap to close first
- choosing whether to finish the current local slice or activate the next already-compatible one

In these cases, the agent must:
1. choose the most coherent option
2. explain the reasoning briefly
3. execute the bounded slice
4. validate it
5. continue autonomously

Stop only for higher-order forks, such as:
- architecture boundary changes
- durable semantic model changes
- scope expansion into a new workstream
- materially different long-term strategic directions
- conflicts between frozen sources that cannot be resolved locally

## Forced local-fork autonomy rule

This is a hard rule.

Once the active direction of a workstream is established, the agent must not stop for local sequencing or bounded implementation forks that remain:
- within the approved architecture direction
- within the active phase/workstream
- within the approved constraints
- within the accepted quality bar

Repeatedly stopping on local forks is incorrect autonomous behavior.

The correct behavior is:
- decide
- execute
- validate
- continue
- then report clearly

## Autonomous execution and checkpoint discipline

For non-trivial work, the agent is expected to manage execution professionally and autonomously.

When useful, the agent should:
- create an isolated branch
- use the canonical repo checkout for low-risk, documentation, and accepted-baseline work
- create an additional git worktree when the benefit is concrete and cleanup safety is clear
- use an isolated worktree by default for risky runtime/performance/redesign families when the redesign policy requires it
- keep the working tree focused
- use logical commits/checkpoints at meaningful validated boundaries
- keep durable progress files updated during execution

Default expectation:
- documentation, control-pack, and low-risk bounded edits may stay in the canonical checkout
- risky redesign paths should use explicit isolation and checkpointing
- trivial scoped edits may stay in the current branch if clean and isolated
- extra git worktrees are required for risky redesign families and optional otherwise

The agent must not use branch/worktree/commit mechanics casually or decoratively.

It should use them when they improve:
- traceability
- rollback safety
- phase isolation
- reviewability
- freeze confidence

If a step is intended to become frozen, the agent should leave it in a state that is:
- isolated
- validated
- checkpointable
- easy to review

## Step-freeze execution rule

Each meaningful step should be treated as a candidate frozen step.

The agent should work as if each step may become a durable baseline for later work.

Therefore each step should aim to be:
- internally coherent
- strongly validated
- bounded in scope
- reviewable from evidence
- safe to build upon without immediately reopening it

Do not leave a step half-shaped if it is being presented for review.

## Persistent context update rule

For any non-trivial completed step, update the relevant persistent project files before stopping.

At minimum, when applicable:
- update `docs/execution/STATUS.md`
- update the active milestone `TASKS.md` and `RISKS.md`
- append to `docs/execution/WORKLOG.md`
- append to `docs/execution/DECISIONS.md` for meaningful decisions
- update `docs/execution/FREEZE.md` when freeze classification changes

Do not leave important execution state only in chat output.  
The repository files are the durable working memory.

## Reporting discipline

Final reports must be:
- evidence-rich
- audit-friendly
- token-efficient
- sufficient for supervisor review without routine follow-up

Every non-trivial report must include:

1. Executive verdict
2. Scope touched
3. Boundary compliance
4. Implementation summary
5. Critical code evidence
6. Validation commands and outcomes
7. Drift / risk check
8. Open issues
9. Next step

Do not pad reports with generic claims.  
Do not restate the prompt.  
Do not repeat the same evidence twice.  
Use the smallest sufficient code excerpts.

## Token-efficiency rule

Actively control token use, but never by lowering rigor.

Do not save tokens by:
- skipping validation
- omitting critical evidence
- weakening investigation
- hiding uncertainty
- reducing implementation quality

Save tokens by:
- removing repetition
- using compact structured sections
- using short direct sentences
- grouping similar changes
- citing exact files/symbols instead of broad paraphrase
- showing only load-bearing code excerpts
- compressing wording without compressing meaning

Optimize report size, not thinking quality.

## Chained autonomous execution rule

For this repository, autonomous execution should proceed in bounded milestone chains, not in single-step confirmation loops.

The agent is authorized to:
- close a completed phase formally
- activate and execute the next already-approved local milestone from the active control files
- continue through compatible local milestones without asking for confirmation at every phase boundary

Do not stop merely because a step becomes review-cleared.
Do not stop for local sequencing forks.
Do not stop for bounded next-step choices already implied by the active roadmap, `STATUS.md`, and active milestone files.

Continue autonomously while:
- the work remains within the approved architecture direction and accepted implemented baseline
- the work remains within the approved roadmap/milestone chain
- no strategic fork appears
- no unresolved blocker remains
- confidence is at least 90% after investigation and validation

If confidence drops below 90%, investigate and validate further instead of guessing.
Stop only if confidence still cannot be raised without:
- a strategic decision
- a scope expansion
- or an architectural change

The correct stopping points are:
- meaningful validated milestone-chain completion
- real blocker
- strategic fork
- or explicit boundary of the approved execution chain

## Working mode after freeze

Official current status:

**Current accepted control baseline under active redesign**

Interpretation:
- the current code baseline is trustworthy and production-freezeable
- the live architecture direction is set in `docs/execution/` and `docs/refactor/`
- implementation work is active only inside the current approved milestone/workstream and adjacent approved hardening or bugfix scope
- risky runtime/performance/redesign work must use isolated worktrees/branches and promote only minimal validated diffs
- future changes must stay inside active milestone execution, approved architecture-definition work, bugfixes, hardening, or narrowly justified operational completion
- do not casually reopen closed families and do not let stale pre-rerank notes steer the repo

## GPT-5.4-class autonomy expectations

This repository expects strong autonomous agents to be capable of:
- reading the repo before acting
- consulting authoritative docs before deciding
- creating the minimum sufficient evidence
- making bounded decisions
- implementing narrow slices
- validating thoroughly
- continuing independently
- and stopping only for real reasons

The expected default posture is:
- autonomous
- evidence-first
- disciplined
- architecture-safe
- honest
- and relentless about clean forward progress

For MLXs redesign work, strong autonomy also means:
- not fragmenting one well-understood front into long micro-family chains
- escalating to integrated refactor or closure/rerank once the evidence says the front is understood
- treating specs and docs as governance inputs that can be updated after redesign, not cages that force stale architecture
- considering built-in MLX capabilities, custom extensions, and custom Metal kernels only after the bottleneck is proven to be substrate-level

If the next step is obvious, do it.  
If the next step is not obvious, investigate.  
If a real stop condition is not present, do not stop.
