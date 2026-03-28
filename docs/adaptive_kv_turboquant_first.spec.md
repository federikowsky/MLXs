# Adaptive KV TurboQuant-First Resident Architecture Specification

## 1. Scope and Intent

This document is the canonical source of truth for the **completed TurboQuant-first** Adaptive KV branch in MLXs.

It records the final retained architecture, validation stance, and archival position for the branch that replaced the earlier operational tier model:

- `FULL`
- `COMPRESSED`
- `EVICTED`

with a resident-profile model:

- `TQ_SAFE`
- `TQ_AGGR`
- `EVICTED`

The retained implementation is grounded in the current repository structure:

- generic semantic core under `src/mlxs/adaptive_kv/`,
- explicit runtime-family layer under `src/mlxs/adaptive_kv/families/`,
- model-specific adapters under `src/mlxs/adaptive_kv/adapters/`,
- generation orchestration under `src/mlxs/generate/`,
- family-aware support gating through runtime capabilities and adapter selection.

This specification does not define a compatibility-preserving migration from the older `FULL / COMPRESSED / EVICTED` baseline. It defines the replacement architecture that was implemented and retained for this branch.

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

The final retained implementation adds:

- persistent exact execution slabs as the resident memory substrate,
- persistent execution packs and execution-pack views as the retained serving substrate above those slabs,
- exact pack-native streaming attention,
- cold-batch / deferred-compaction mutation handling,
- delayed observer flushes at policy boundaries,
- executor and usage-sync improvements that materially strengthened the serving path while preserving exactness.


## 2. Design Goals

The branch implementation defined by this specification must satisfy all of the following goals.

### 2.1 TurboQuant-first resident design

TurboQuant remains the primary resident backend contract and naming surface for this branch. In the retained implementation, however, TurboQuant-first means **resident-profile planning plus a backend-owned execution substrate**, not a retained lossy resident-compression path.

Resident KV state is managed as profiled resident execution slabs and slice views. The earlier lossy resident backend direction was rejected because it could not satisfy oracle fidelity honestly.

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
- Exact resident profile intended for ordinary steady-state operation.
- Backed in the retained implementation by larger persistent execution slabs and the more conservative resident policy.
- Preferred profile for important or recently active resident blocks.

`TQ_AGGR`
- Exact resident profile intended for more aggressive resident-memory management.
- Backed in the retained implementation by smaller persistent execution slabs and the more aggressive resident policy.
- Used when a block should remain resident but no longer warrants `TQ_SAFE`.

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

### 4.4 Resident profile is not one fixed execution path

Resident profile does not imply one fixed attention path. Execution correctness is defined relative to the appropriate family baseline, not relative to a backend-native shortcut.

In the retained implementation, both resident profiles are execution-ready through persistent execution packs built over the exact slab substrate. The execution-mode contract remains at the runtime boundary, but the retained hot resident path is not a lossy quantize/dequantize loop.


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

Storage representation and execution mode are separate decisions at the contract level.

The retained implementation realizes that separation through:

- exact resident storage in persistent execution slabs,
- persistent execution packs and ordered execution-pack views above those slabs,
- one-pack exact fast paths where available,
- exact pack-native streaming attention for multi-pack resident views.

The backend therefore exposes execution views separately from logical block ownership and replay provenance, even though the retained hot path is exact and execution-native.

### 6.5 TurboQuant as the primary backend

TurboQuant is the required resident backend for this branch.

For this branch, TurboQuant is defined as:

- a new MLX-aligned resident backend architecture,
- not identical to the earlier `COMPRESSED` tier implementation,
- not merely a rename of `AdaptiveCompressedRunStore`,
- and, in the retained branch, concretely implemented as a persistent exact execution fabric.

The retained backend keeps the resident-profile architecture and backend-owned contracts, but the branch’s final excellence bar was achieved through the exact slab/slice execution-substrate direction rather than through a retained lossy TurboQuant resident compression path.

### 6.6 Final retained backend shape

The final retained backend consists of:

- `ResidentBlockHandle` objects that carry block metadata plus slab fragments,
- persistent exact execution slabs with profile-specific slab policy as the memory substrate,
- persistent execution packs and execution-pack views as the retained serving substrate above those slabs,
- Family B window-visible execution-pack clipping over the retained resident logical span,
- Family C strict KV-versus-recurrent separation,
- exact pack-native streaming attention,
- cold-batch / deferred-compaction mutation handling,
- executor and usage-sync improvements retained on top of the slab/execution-pack substrate.


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

`ExecutionSliceRef`
- Execution-facing slice metadata over persistent resident slabs with family/profile-specific visibility semantics.

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

### 12.4 Final hardening matrix

The final hardening/proof pass for this branch broadened validation beyond the retained representative closure set.

That final matrix covered:

- Family A, Family B, and Family C,
- short and long prompt/decode shapes,
- true soft-pressure and true hard-pressure operating points,
- one warmup run plus three measured runs per core case,
- additional pathology probes for equal-budget edges, tight hard-churn regimes, and chunk-size sensitivity.

The core repeated matrix contained twelve measured cases:

- Family A: short soft, short hard, long soft, long hard
- Family B: short soft, short hard, long soft, long hard
- Family C: short soft, short hard, long soft, long hard

Final measured outcome:

- all measured matrix cases remained exact,
- `reference_token_match` remained closed,
- `first_mismatch_token` remained `null`,
- resident-byte outcomes were stable within each repeated case,
- topology rebuild counts were stable within each repeated case.

Representative medians from the broadened repeated matrix were:

- Family A: short soft `277.82 tok/s`, short hard `193.20 tok/s`, long soft `55.76 tok/s`, long hard `48.32 tok/s`
- Family B: short soft `189.70 tok/s`, short hard `150.54 tok/s`, long soft `77.15 tok/s`, long hard `76.39 tok/s`
- Family C: short soft `178.87 tok/s`, short hard `155.79 tok/s`, long soft `80.66 tok/s`, long hard `65.26 tok/s`

These values characterize the retained branch on the tiny-model proof harness used for exact closure. They are branch-proof evidence, not a promise of universal production throughput.

### 12.5 Pathology findings and retained caveats

The final pathology hunt did not find a new correctness blocker or a support-honesty blocker.

Important retained findings:

- equal-budget edge cases remained exact and converged into the expected hard-pressure regime with bounded resident bytes and explicit replay,
- tight hard-churn regimes remained exact and operationally stable, but throughput dropped materially as pressure churn increased,
- very small prefill chunks remained exact but were materially slower because they increased policy-window cadence and execution-view rebuild cadence,
- short soft-pressure runs on the tiny-model proof harness showed more variance than long or hard runs; that variance did not correlate with correctness loss, resident-byte drift, or pressure-state instability.

Operational caveat:

- the retained branch is exact under small chunk sizes and churn-heavy hard budgets, but those settings are not performance-neutral; future evaluations should treat chunk-size selection as an operational parameter, not as a purely cosmetic benchmark knob.


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


## 14. Final Retained Implementation

The completed branch retains the following implementation position.

- Tier-oriented runtime vocabulary was replaced with resident-profile vocabulary.
- `ResidentBackend`, `ResidentProfileDescriptor`, and `ResidentBlockHandle` are the resident-backend boundary.
- Resident execution is backed by persistent exact execution slabs plus persistent execution packs and execution-pack views above them.
- Family A uses the full resident visible history through the execution-pack serving substrate.
- Family B uses explicit resident logical span, effective visible span, and window-visible execution-pack clipping.
- Family C keeps KV-bearing layers inside resident-profile planning and recurrent/state-array layers outside it.
- Replay and recovery remain explicit and exact.
- Metrics and diagnostics are resident-profile-aware and expose both memory-substrate and execution-pack serving-path state.
- No operational `FULL` resident tier or hidden dense runtime fallback remains.

### 14.1 Frozen baseline invariants

The retained branch is frozen around the following invariants.

Control-plane invariants:

- `TQ_SAFE`, `TQ_AGGR`, and `EVICTED` are the only operational resident outcomes.
- `EVICTED -> resident` is never implicit; replay-backed recovery is mandatory.
- support is granted only through explicit runtime-family/adaptor gating and per-family oracle validation.

Data-plane invariants:

- slabs remain the memory truth for residency, replay provenance, and cold mutation bookkeeping,
- execution packs remain the retained serving truth above the slab substrate,
- transient grouped-segment assembly is not the retained hot path,
- no retained lossy resident compression path is allowed to claim support without renewed oracle closure.

Hot-path invariants:

- one-pack exact fast paths are allowed where validated,
- multi-pack resident execution remains exact relative to the family baseline,
- no hidden dense fallback may silently replace the retained resident execution path.

Cold-path invariants:

- replay and recovery remain explicit,
- cold mutation batching and deferred compaction may reduce cost but must not change semantics,
- compaction is an implementation detail, not a correctness requirement.

Family invariants:

- Family A remains exact against contiguous full-history semantics,
- Family B remains exact against resident logical span plus effective visible span under window-local replay semantics,
- Family C keeps KV-bearing layers inside resident-profile planning and recurrent/state-array layers outside it.

### 14.2 Forbidden regressions and future validation obligations

Future changes must not reintroduce:

- an operational `FULL` resident tier,
- a hidden dense runtime fallback,
- a retained lossy resident backend path without new oracle closure,
- transient grouped-segment execution as the normative hot path,
- benchmark-negative view-repair/composition paths or grouped-query fallbacks merely because they look architecturally elegant.

Future changes that touch policy hot paths, resident execution, or replay behavior must rerun:

- per-family short/long validation,
- soft/hard budget validation,
- representative warm repeated sweeps,
- budget-edge and churn/pathology checks,
- chunk-size sensitivity checks when prefill/decode cadence changes materially.

## 15. Final Closure Status

The branch is complete and retained because all of the following are now true.

- No operational `FULL` resident tier remains in runtime design.
- `TQ_SAFE`, `TQ_AGGR`, and `EVICTED` are the only operational resident outcomes.
- The semantic core remains representation-agnostic and family-aware.
- Family A, Family B, and Family C support are oracle-validated within the retained support envelope.
- Family B semantics are implemented with explicit resident logical span, effective visible span, window-local replay semantics, and window-visible slice execution.
- Family C recurrent/state-array layers remain outside resident-profile planning, degradation, and transitions.
- Replay and recovery remain explicit and exact.
- Fidelity is closed.
- The long path improved materially over earlier retained baselines.
- The execution-pack serving path above the slab memory substrate is the current retained serving baseline.
- Product-level evaluation supports Adaptive KV primarily as an operational serving architecture rather than a guaranteed conversational-quality improver.
- A narrow Family C short-hard hybrid sampled-serving caveat remains.

### 15.1 Support and capability freeze

For this branch, "supported" means:

- the model path matches a registered runtime-family adapter,
- the runtime surface is inside the single-request generation envelope,
- all three operational resident outcomes (`TQ_SAFE`, `TQ_AGGR`, `EVICTED`) preserve exact family semantics,
- the path is inside the retained oracle-validated support envelope.

The final frozen support matrix is:

- Family A `full_kv`:
  supported for the registered Llama-style full-attention adapter when the runtime uses the standard homogeneous `list[KVCache]` baseline, the layers remain full-attention, and `head_dim % 32 == 0`
- Family B `full_kv`:
  supported only when the registered Ministral3 path is assessed as a homogeneous full-attention `list[KVCache]` baseline rather than the explicit sliding/windowed runtime
- Family B `windowed_kv`:
  supported for the registered Ministral3 adapter when sliding layers use the windowed runtime contract and the cache layout matches the adapter's expected `RotatingKVCache`/`KVCache` split; `head_dim % 32 == 0` remains required
- Family C `full_kv`:
  supported only when the registered Qwen3.5 path is assessed as a homogeneous `list[KVCache]` full-attention baseline
- Family C `hybrid_state`:
  supported for the registered Qwen3.5 adapter when KV-bearing layers and linear/state-array layers match the hybrid adapter contract (`KVCache` on full-attention layers and `ArraysCache` on linear-attention layers); `head_dim % 32 == 0` remains required

Explicitly unsupported surfaces remain:

- `compile_decode=True`
- legacy `quantized_kv_start > 0`
- external cache reuse
- multimodal or `input_embeddings` requests
- missing or malformed `model.make_cache()`
- unregistered adapters or families
- batch/speculative surfaces and complex cache-family compositions outside the registered support envelope

### 15.2 Product-level E2E position

The longer-form product-grade E2E revalidation for the current retained execution-pack baseline used the strongest realistic chat-capable checkpoint that was practically available in the environment:

- `unsloth/Llama-3.2-1B-Instruct` on Family A.

Family C was omitted from the main product-grade matrix because the supported `qwen3_5` checkpoints available in the environment did not behave like credible product-chat baselines. Family B was omitted because no practical chat-capable Ministral-family checkpoint was staged locally for a serious long-form product comparison.

The long-form E2E regimes were meaningfully separated by projected baseline full-KV footprint relative to the configured Adaptive hard budget:

- comfortable mean ratio: `0.58x`
- borderline mean ratio: `1.94x`
- stressed mean ratio: `4.71x`

Observed product-level conclusion:

- Adaptive KV remained slower than normal baseline inference in all three regimes,
- no clear user-visible conversational quality win was established on the evaluated checkpoint,
- the primary retained value remained operational: explicit pressure management, evictions, recomputes, replay-backed continuation, and bounded serving behavior under tighter envelopes.

Operationally, the long-form run recorded:

- comfortable: `0` evictions / `0` recomputes,
- borderline: `174` evictions / `21` recomputes,
- stressed: `580` evictions / `24` recomputes.

Operational caveats retained:

- end-of-turn `resident_bytes` can overshoot the nominal hard budget in longer borderline and stressed runs, so the retained system should be described as pressure-managed rather than as a strict hard-cap proof from final snapshots,
- the narrow Family C short-hard hybrid sampled-serving caveat remains part of the retained support and performance story.

## 16. Historical Risks and Rejected Paths

### 16.1 Lossy resident TurboQuant path

Historical risk:
- a resident-profile combination could be operationally stable but fail oracle validation due to lossy resident representation fidelity.

Final retained handling:
- the lossy resident backend direction was rejected.
- the retained backend is the exact slab/slice execution substrate.

### 16.2 Transient grouped-segment execution path

Historical risk:
- repeated grouped-segment assembly created avoidable hot-path cost and excessive visible fragmentation.

Final retained handling:
- transient grouped segments were rejected in favor of persistent slabs and slice-based resident views.

### 16.3 View-query / composition-only optimizations

Historical risk:
- several local view-repair and composition-only optimizations looked plausible but did not move the real serving bottleneck.

Final retained handling:
- benchmark-negative view-query/composition optimizations were rejected rather than retained as complexity.

### 16.4 Family C grouped-query segmented fallback

Historical risk:
- the remaining short Family C gap suggested a grouped-query-specific fallback might still be needed.

Final retained handling:
- that fallback was benchmarked, rejected, and not retained.
- the current retained execution-pack baseline keeps a narrow short-hard hybrid caveat rather than relying on that fallback.

### 16.5 Family B / Family C semantic boundaries

Historical risk:
- Family B could collapse into masked Family A semantics, or Family C recurrent/state-array layers could leak into resident-profile planning.

Final retained handling:
- both boundaries remain explicit, validated, and retained.

### 16.6 Truthfulness requirement

Final archival caveat:
- the branch achieved excellence through the retained slab/slice execution-substrate direction, not through a retained lossy TurboQuant resident compression path.
- future work must not overclaim compression benefits that are not actually retained.


## 17. Final Recommendation

This document is the canonical source of truth for the completed TurboQuant-first Adaptive KV branch in MLXs.

Future work, if any, should build from this retained baseline:

- resident profiles `TQ_SAFE / TQ_AGGR / EVICTED`,
- persistent exact execution slabs as the memory substrate,
- persistent execution packs and execution-pack views as the retained serving substrate,
- exact pack-native streaming attention,
- explicit replay/recovery,
- Family A / B / C runtime-family separation.

Old `FULL / COMPRESSED / EVICTED` operational assumptions are superseded and should not be reopened. Product-level claims for this retained baseline must stay honest: its strongest evidence is as an exact, pressure-managed serving architecture, not as a universal guarantee of visibly better conversational quality under stress.
