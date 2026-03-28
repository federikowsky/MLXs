# Adaptive KV TurboQuant-First Resident Architecture Specification

## 1. Scope and Intent

This document is the canonical source of truth for the **TurboQuant-first** Adaptive KV redesign branch in MLXs.

It specifies a branch-only architecture that intentionally breaks from the current retained production baseline. The branch redesign replaces the current operational tier model:

- `FULL`
- `COMPRESSED`
- `EVICTED`

with a resident-profile model:

- `TQ_SAFE`
- `TQ_AGGR`
- `EVICTED`

The design is intended for direct implementation inside the existing MLXs Adaptive KV package and generation flows. It is grounded in the current repository structure:

- generic semantic core under `src/mlxs/adaptive_kv/`,
- explicit runtime-family layer under `src/mlxs/adaptive_kv/families/`,
- model-specific adapters under `src/mlxs/adaptive_kv/adapters/`,
- generation orchestration under `src/mlxs/generate/`,
- family-aware support gating through runtime capabilities and adapter selection.

This specification **does not** define a compatibility-preserving migration from the current retained baseline. It defines the replacement architecture for this branch.

The redesign preserves the following principles:

- exact model semantics,
- explicit replay and recovery,
- policy external to attention semantics,
- runtime-family-aware architecture,
- strict support honesty,
- production-oriented implementation constraints.

The redesign intentionally changes the following:

- no operational `FULL` resident tier,
- no requirement to preserve current `BlockTier` runtime semantics,
- no requirement to preserve current compressed-run storage assumptions,
- no requirement to keep current metrics, config naming, or internal handle layout if they encode obsolete tier concepts.


## 2. Design Goals

The branch implementation defined by this specification must satisfy all of the following goals.

### 2.1 TurboQuant-first resident design

TurboQuant is the primary resident representation backend for this branch. Resident KV state is not managed as "full tensors plus an optional compressed tier." Instead, resident KV state is managed as a set of **profiled TurboQuant-backed resident representations**.

### 2.2 No `FULL` operational resident tier

The runtime design must not depend on a full-fidelity resident KV tier as a normal operational state. Validation harnesses and oracle paths may use dense contiguous references, but runtime residency planning must not.

### 2.3 Policy over resident profiles

Adaptive KV must be defined as:

- a policy over logical blocks,
- selecting resident profiles,
- over pluggable resident representations,
- with explicit replay-backed recovery for evicted content.

The policy must not be defined in terms of one storage format's internal mechanics.

### 2.4 Exact semantics

The branch must preserve exact family baseline semantics for every supported family/profile combination. Replay correctness, resident ordering, window semantics, and hybrid-state reconstruction must all remain exact.

### 2.5 Production-oriented design

The branch must be written for a retained production-quality codebase, not as a conceptual experiment. Contracts, responsibilities, validation obligations, and failure handling must be explicit enough to guide a large refactor without ambiguity.

### 2.6 Family-aware architecture

The design must preserve and strengthen the existing runtime-family layer:

- Family A `full_kv`
- Family B `windowed_kv`
- Family C `hybrid_state`

The design must not collapse those families into one pseudo-generic substrate.


## 3. Non-Goals

The following are explicitly out of scope for this branch specification.

- Backward compatibility with the retained `FULL / COMPRESSED / EVICTED` runtime model.
- Gradual migration layers, compatibility shims, or dual-mode runtime operation.
- An operational fallback that silently reintroduces `FULL` resident runtime state.
- Support for unsupported generation surfaces such as `compile_decode`, legacy `quantized_kv_start`, or external cache reuse.
- A promise of support for every future model family merely because the architecture is generic.
- A learned controller, RL policy, semantic retrieval layer, or topic-aware retrieval module.
- Token-level residency planning.
- Attention-semantic modification, approximate replay, or heuristic recovery.


## 4. Conceptual Model

### 4.1 Resident profiles replace runtime tiers

The core operational concept is **resident profile**, not compression tier.

Each logical block is always in exactly one of three residency outcomes:

- `TQ_SAFE`
- `TQ_AGGR`
- `EVICTED`

`TQ_SAFE` and `TQ_AGGR` are both resident states. `EVICTED` is not resident.

### 4.2 Meaning of the profiles

`TQ_SAFE`
- TurboQuant-backed resident representation intended for ordinary steady-state operation.
- Higher-fidelity and/or lower-risk than `TQ_AGGR`.
- Preferred profile under moderate memory pressure where the block remains resident.

`TQ_AGGR`
- TurboQuant-backed resident representation intended for more aggressive memory reduction.
- Lower-memory and/or higher-risk than `TQ_SAFE`.
- Used when the planner decides that a block should remain resident but no longer warrants the safer profile.

`EVICTED`
- No resident representation is kept.
- Only logical metadata, ghost/replay provenance, and recovery bookkeeping remain.
- Any transition from `EVICTED` back to resident state requires explicit replay-backed reconstruction.

### 4.3 Policy versus representation

The policy selects the **target profile** for a block. The resident backend realizes that target through family-correct storage and execution materialization.

This separation is mandatory:

- the planner does not reason about raw quantized tensors,
- the backend does not decide policy,
- the attention path does not redefine policy semantics,
- recovery is explicit and family-correct.

### 4.4 Storage profile is not execution mode

Resident profile does not imply one fixed attention path. A profile may be stored in one format and executed through another materialization strategy. Execution correctness is defined relative to the appropriate family baseline, not relative to a backend-native shortcut.


## 5. Core Architecture

The generic semantic core remains under `src/mlxs/adaptive_kv/` and must remain representation-agnostic.

The core owns:

- logical block metadata,
- block chronology and source-span mapping,
- score computation,
- usage aggregation,
- pressure classification,
- protections and pinning,
- ghost metadata,
- replay triggering,
- profile planning,
- transition bookkeeping,
- metrics and diagnostics,
- top-level orchestration around policy windows and recovery barriers.

The core must **not** own:

- TurboQuant storage tensor layout,
- family-specific attention execution details,
- family-specific replay materialization details,
- backend-native quantization shortcuts.

The existing separation between:

- semantic core,
- runtime-family layer,
- model adapter layer,
- capability assessment,

remains valid and should be retained. This branch changes the resident model inside that platform shape, not the existence of the shape itself.


## 6. Resident Representation Architecture

### 6.1 Resident backend abstraction

The branch must introduce a `ResidentBackend` contract that owns resident representation mechanics for supported profiles.

At minimum, that contract must support:

- create resident representation for a logical block under a requested profile,
- change profile for a resident block,
- remove resident representation on eviction,
- expose memory cost and profile descriptors,
- expose a family-correct execution view or materialization handle,
- reconstruct resident state from replayed authoritative tokens,
- report profile-specific runtime capabilities.

`ResidentBackend` is the primary owner of storage and materialization mechanics. The policy core sees declared costs and supported transitions, not raw tensors.

### 6.2 Resident block handle abstraction

The branch must introduce `ResidentBlockHandle` as the canonical opaque handle for resident logical blocks.

Required properties:

- profile identity,
- logical span identity,
- backend ownership,
- sufficient information to produce family-correct execution materialization,
- stable linkage to replay provenance and diagnostics.

The exact granularity beyond block-level is implementation-defined. Run coalescing is allowed and expected where profitable, but the specification does **not** require a separate `ResidentRunHandle`. Internal coalescing may be used so long as:

- block ownership remains recoverable,
- per-block costs remain measurable,
- family-correct visibility semantics remain enforceable,
- promotion/demotion/eviction semantics remain exact.

### 6.3 Resident profile abstraction

The branch must define `ResidentProfile` and `ResidentProfileDescriptor`.

Each profile descriptor must state:

- profile name,
- backend support,
- estimated memory cost class,
- estimated execution cost class,
- estimated replay avoidance benefit,
- fidelity or risk class,
- allowed inbound/outbound transitions,
- permitted execution modes per family.

### 6.4 Storage profile versus execution mode

Storage representation and execution mode are separate decisions.

Examples of acceptable relationships:

- a `TQ_SAFE` block may remain quantized in storage but materialize to dense tensors for execution;
- a `TQ_AGGR` block may execute via direct compressed attention if validated;
- the same profile may use different execution modes in different families.

The backend must therefore expose **execution materialization** separately from storage state.

### 6.5 TurboQuant as the primary backend

TurboQuant is the required resident backend for this branch.

For this branch, TurboQuant is defined as:

- a new MLX-aligned resident backend architecture,
- compatible with the repo's existing quantize/dequantize precedent,
- but not identical to the retained `COMPRESSED` tier implementation,
- and not merely a rename of current `AdaptiveCompressedRunStore`.

The first implementation may ship only the TurboQuant backend, but the branch architecture must preserve backend-pluggable contracts so that resident policy remains representation-agnostic.


## 7. Execution Semantics

### 7.1 Normative separation

Storage representation is independent from attention execution policy.

This is a hard architectural rule. The backend may store resident state in one form and execute it through another, provided the resulting behavior is exact relative to the appropriate family baseline.

### 7.2 Allowed execution modes

A resident profile may execute through any of the following modes, depending on the validated family/profile combination:

- direct compressed attention,
- dequantize/materialize before attention,
- family-specific execution path that is neither of the above, provided it is exact.

Execution mode is selected **per family/profile combination**, not globally.

### 7.3 Correctness definition

Correctness is defined against the relevant family baseline semantics:

- contiguous full-history attention semantics for Family A,
- exact sliding/windowed baseline semantics for Family B,
- exact hybrid-state baseline semantics for Family C.

Correctness is **not** defined against the elegance or purity of a backend-native compressed attention shortcut.

### 7.4 Design priority

The branch takes the following position:

- direct compressed attention is optional,
- dequantize-on-read is acceptable,
- family-specific materialization is acceptable,
- exact semantics and validated fidelity take priority over compressed-attention elegance.

No family/profile combination may be considered supported merely because a direct compressed path exists. It must be validated against the correct family oracle.


## 8. Runtime Family Integration

### 8.1 Shared rule

Runtime-family integration remains explicit. Model adapters classify the runtime and bind into family-specific resident bindings. They do not secretly own the resident substrate.

Profile support must be assessed per family and per family/profile combination.

### 8.2 Family A: `full_kv`

Family A remains the simplest runtime family and the first implementation target.

Normative properties:

- token-addressable resident history,
- TurboQuant-backed resident profiles for every resident block,
- exact replay from authoritative source token spans,
- resident execution defined against the family's contiguous full-history baseline,
- no operational dense resident fallback.

Supported operational outcomes:

- `TQ_SAFE`
- `TQ_AGGR`
- `EVICTED`

Family A remains the primary oracle and regression gate for the branch implementation, but it is **not** the universal substrate that other families inherit by default.

### 8.3 Family B: `windowed_kv`

Family B is first-class and must not be treated as "Family A plus a different mask."

The branch implementation must define the following Family B concepts explicitly:

**resident logical span**
- the logical token interval owned by a resident block/profile in the policy core.

**effective visible span**
- the subset of the resident logical span that is actually visible to a given layer/query under sliding/window semantics.

**window-local replay semantics**
- replay and recovery are defined against the exact sliding/windowed family baseline, not against a full-history baseline with a window mask attached after the fact.

**window-aware execution materialization**
- execution views must be materialized with family-correct visibility semantics, not by naively reusing Family A resident assembly.

Normative Family B rules:

- Family B remains token-addressable, but execution and recovery are windowed.
- Profile support is valid only when TurboQuant-backed resident state can execute exactly under windowed visibility semantics.
- Recovery must rebuild family-correct windowed runtime state.
- Family B replay must not be specified as trivial Family A replay plus mask substitution.

### 8.4 Family C: `hybrid_state`

Family C must maintain a strict planning boundary.

KV-bearing layers are:

- inside the resident profile planner,
- inside the resident degradation chain,
- inside resident profile transitions.

Non-KV recurrent/state-array layers are:

- outside the resident profile planner,
- outside the resident degradation chain,
- outside resident profile transitions.

Those same recurrent/state-array layers remain:

- inside replay correctness,
- inside runtime reconstruction,
- inside exact family baseline semantics.

Normative Family C rules:

- recurrent/state-array layers must never be mixed into the same profile-planning semantics as KV-bearing resident blocks;
- replay must reconstruct both:
  - resident-profile-managed KV state for KV-bearing layers,
  - exact pass-through recurrent/state state for non-KV layers;
- any family binding that cannot preserve this separation must reduce support honestly.


## 9. Policy and Transition Model

### 9.1 Operational transitions

The canonical transition model for this branch is:

- `TQ_SAFE -> TQ_AGGR -> EVICTED`

Allowed promotion:

- `TQ_AGGR -> TQ_SAFE`

Disallowed shortcut:

- no `EVICTED -> resident` without replay-backed recovery.

### 9.2 Planning before realization

The core must explicitly separate:

1. deciding the target profile for a block,
2. realizing that transition through backend operations.

This prevents policy from becoming entangled with one backend's internal storage mechanics.

### 9.3 Hysteresis and anti-thrashing

The branch must retain and adapt existing anti-thrashing principles:

- minimum dwell per resident profile,
- promotion cooldown,
- demotion cooldown,
- recent-tail protection,
- structural protection or pinning,
- replay-aware stabilization,
- profile-aware hysteresis thresholds.

### 9.4 Ghost metadata

Ghost metadata remains required after eviction. At minimum it must retain:

- block identity,
- authoritative source span,
- last resident profile,
- score-at-eviction or equivalent planning signal,
- last eviction step,
- reuse/reactivation indicators if the planner uses them.

### 9.5 Profile planning

The planner must decide among resident profiles using:

- observed usage,
- structural priors,
- pressure state,
- protection state,
- declared profile costs,
- replay cost awareness,

without depending on raw backend-native details.


## 10. Data Model and Contracts

The implementation for this branch should introduce or refactor toward the following internal contracts.

### 10.1 Required concepts

`ResidentProfile`
- Enum for `TQ_SAFE`, `TQ_AGGR`, `EVICTED`.

`ResidentProfileDescriptor`
- Declares per-profile costs, fidelity/risk class, execution-mode allowances, and allowed transitions.

`ResidentBackend`
- Owns resident representation creation, profile conversion, eviction-time teardown, replay reconstruction, and execution materialization.

`ResidentBlockHandle`
- Opaque per-block resident handle owned by a backend.

`ResidencyPlanner`
- High-level policy planner that chooses target profiles under pressure, protections, and score state.

`ProfileTransitionPlanner`
- Applies transition legality, hysteresis, dwell, cooldown, and anti-thrashing rules before backend realization.

`FamilyResidentBindings`
- Family descriptor plus family-specific resident backend, replay strategy, and execution semantics declarations.

`ResidentStateView`
- Ordered attention/runtime view assembled for execution from family-correct resident handles.

`ResidentAttentionSegment`
- Execution-facing segment metadata with family/profile-specific visibility semantics.

`FamilyProfileCapabilities`
- Support descriptor for each family/profile combination, including oracle-validation requirement and permitted execution modes.

### 10.2 Superseded current concepts

The following current concepts are superseded on this branch and should not be preserved as branch-defining runtime contracts:

- `BlockTier` as the operational runtime abstraction,
- `FULL` versus `COMPRESSED` resident semantics,
- any requirement that dense resident slices exist operationally,
- any assumption that compressed-run storage is itself the architecture.

Coalesced run storage remains allowed as an implementation optimization, but it is not the branch's primary conceptual contract.


## 11. Cost Model

The policy must reason using declared cost signals, not backend internals.

### 11.1 What the policy may know

Per family/profile combination, the planner may consume:

- estimated memory bytes,
- estimated execution cost class,
- estimated replay/recovery cost,
- fidelity or risk class,
- family-specific execution penalty multipliers if measured and declared,
- transition cost class if needed for anti-thrashing.

### 11.2 What the policy must not know

The planner must not depend on:

- raw quantized tensor layout,
- MLX kernel-specific storage details,
- backend-internal buffering schemes,
- hidden execution shortcuts.

### 11.3 Fidelity/risk class

The cost model may include an explicit fidelity or risk class so that:

- `TQ_SAFE` can be preferred for important or recently active blocks,
- `TQ_AGGR` can be used when memory pressure dominates but the block still remains resident,
- support can be reduced if a family/profile combination cannot meet oracle obligations.


## 12. Validation and Oracle Strategy

Removing operational `FULL` from runtime design does **not** remove the obligation to validate against dense family baselines.

### 12.1 Oracle location

Full contiguous baseline behavior is an oracle and harness concern only. It is not an operational resident tier.

Validation may use:

- family baseline non-adaptive generation,
- family-correct dense oracle execution,
- replay-focused recovery oracles,
- dedicated regression harnesses.

### 12.2 Per-family/profile validation obligation

For every supported family/profile combination, the implementation must provide:

- explicit oracle validation against the appropriate family baseline, or
- honest reduction of capability/support if oracle closure cannot be achieved.

Support must **not** be granted based only on:

- architectural cleanliness,
- replay-safety alone,
- operational stability alone,
- "close enough" output similarity without documented downgraded capability.

### 12.3 Family-specific oracle rules

Family A:
- validate against dense contiguous full-history family baseline.

Family B:
- validate against exact windowed/sliding family baseline semantics, including visibility and replay correctness.

Family C:
- validate against exact hybrid baseline semantics, including both KV-bearing and pass-through recurrent/state layers.


## 13. Performance and Optimization Principles

The branch implementation must follow these rules.

### 13.1 Algorithmic design before micro-optimization

Choose the correct architecture first:

- explicit planner/backend separation,
- family-correct materialization,
- profile-aware cost model,
- replay correctness.

Do not distort semantics to preserve a micro-optimization.

### 13.2 Minimal allocation and copy discipline

The implementation should minimize:

- repeated dequantize/materialize churn,
- unnecessary resident-state reconstruction,
- redundant host/device synchronization,
- unnecessary copying across profile transitions.

### 13.3 Hot-path awareness

Execution materialization must be designed with decode hot paths in mind, but correctness still comes first. Family-specific optimization is allowed where exactness is preserved and validated.

### 13.4 Storage and execution separation

Storage format must not dictate execution semantics. A backend may choose the fastest validated execution strategy for a given family/profile combination.

### 13.5 No unnecessary abstraction overhead

Abstractions must correspond to real responsibility boundaries:

- semantic core,
- resident backend,
- family execution semantics,
- replay/recovery,
- capability assessment.

Avoid framework-like generality that adds runtime overhead without implementation value.


## 14. Implementation Plan

The later coding run should implement this branch in the following tranches.

### Tranche 1 — Core vocabulary and contract replacement

- Replace tier-oriented core vocabulary with resident-profile vocabulary.
- Refactor runtime contracts, config, metrics, diagnostics, and block metadata away from `FULL / COMPRESSED / EVICTED`.
- Preserve generic semantic-core responsibility boundaries.

### Tranche 2 — Resident backend and handle layer

- Introduce `ResidentBackend`, `ResidentProfileDescriptor`, and `ResidentBlockHandle`.
- Replace direct compressed-run assumptions with backend-owned resident handles.
- Keep internal coalescing implementation-defined.

### Tranche 3 — Family A TurboQuant-first substrate

- Implement TurboQuant-backed Family A resident representation.
- Remove any operational FULL resident path.
- Establish Family A oracle validation as the first hard gate.

### Tranche 4 — Execution-mode layer

- Introduce explicit execution-mode selection per family/profile combination.
- Support dequantize-on-read and family-specific execution materialization as first-class paths.
- Treat direct compressed attention as optional and validation-gated.

### Tranche 5 — Family B implementation

- Implement resident logical span and effective visible span semantics.
- Implement window-local replay semantics.
- Implement window-aware execution materialization.
- Validate each supported Family B profile combination against Family B oracle baselines.

### Tranche 6 — Family C implementation

- Implement strict KV-versus-recurrent planning separation.
- Keep non-KV recurrent/state layers outside resident-profile planning.
- Implement family-correct replay and runtime reconstruction for both KV-bearing and recurrent layers.
- Validate each supported Family C profile combination against Family C oracle baselines.

### Tranche 7 — Metrics, capabilities, and diagnostics

- Rework metrics and diagnostics around resident profiles.
- Add family/profile-aware capability reporting.
- Encode oracle-validation obligations into support status.

### Tranche 8 — Cleanup and removal of obsolete assumptions

- Remove obsolete FULL/COMPRESSED runtime assumptions.
- Remove code paths that rely on dense resident fallback.
- Remove old tier semantics from branch runtime logic.


## 15. Acceptance Criteria

The branch implementation is complete only when **all** of the following are true.

- No operational `FULL` resident tier remains in runtime design.
- `TQ_SAFE`, `TQ_AGGR`, and `EVICTED` are the only operational resident outcomes.
- The semantic core remains representation-agnostic and family-agnostic.
- Family A support is oracle-validated against the appropriate dense family baseline.
- Any supported Family B profile combination is oracle-validated against Family B baseline semantics.
- Any supported Family C profile combination is oracle-validated against Family C baseline semantics.
- If oracle closure is not achieved for a family/profile combination, support is reduced honestly.
- Family B semantics are implemented with explicit resident logical span, effective visible span, window-local replay semantics, and window-aware execution materialization.
- Family C recurrent/state-array layers remain outside resident-profile planning, degradation, and profile transitions.
- Family C replay and runtime reconstruction remain exact for both KV-bearing and recurrent/state layers.
- Replay and recovery remain explicit and exact.
- Policy remains external to attention semantics.
- Capability reporting is both family-aware and profile-aware.
- Metrics and diagnostics reflect resident profiles rather than obsolete tier names.


## 16. Risks and Failure Modes

### 16.1 TurboQuant fidelity risk

Risk:
- a family/profile combination may be operationally stable but fail oracle validation due to resident representation fidelity.

Required handling:
- reduce support honestly or restrict the profile/execution-mode combination.

### 16.2 Family B visibility mismatch

Risk:
- incorrect visible-span handling could silently turn Family B into a masked Family A approximation.

Required handling:
- explicit Family B execution materialization contracts and Family B oracle validation.

### 16.3 Family C planning boundary violation

Risk:
- recurrent/state-array layers might accidentally be pulled into profile planning semantics.

Required handling:
- hard contract separation and dedicated Family C validation.

### 16.4 Replay reconstruction mismatch

Risk:
- resident blocks may be replayed correctly while family-specific runtime state is not fully reconstructed.

Required handling:
- family-aware replay backends and per-family recovery oracles.

### 16.5 Transition thrash

Risk:
- poor hysteresis or transition cost modeling may cause resident-profile oscillation.

Required handling:
- dwell rules, cooldowns, replay-aware stabilization, and explicit transition planning.

### 16.6 Abstraction overhead

Risk:
- over-general resident-handle/backend abstraction could create hot-path overhead without value.

Required handling:
- keep contracts narrow, keep coalescing implementation-defined, and optimize by responsibility boundary rather than by framework expansion.


## 17. Final Recommendation

This document is intended to become the canonical source of truth for the TurboQuant-first Adaptive KV redesign branch in MLXs.

Later implementation prompts should treat this specification as decision-complete. Where current retained documents, code paths, or metrics encode `FULL / COMPRESSED / EVICTED` operational semantics, this branch specification supersedes them.
