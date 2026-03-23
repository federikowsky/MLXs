Adaptive KV for Multi-Turn LLM Inference in MLXs

A Production-Serious Observer-Only Adaptive Residency Policy

Abstract

Long multi-turn inference stresses transformer KV-cache memory because standard append-only retention preserves all past context uniformly, while fixed sliding windows bound memory by discarding history without regard to actual usefulness. This work presents a production-serious adaptive KV-cache policy integrated into MLXs that preserves attention semantics while dynamically managing cache residency through block-level transitions among FULL, COMPRESSED, and EVICTED states. The system is training-free, inference-only, and observer-only: it never modifies queries, keys, values, or logits, and instead acts exclusively at the memory-management layer.

The final design combines:
	•	block-level usage observation,
	•	bounded multi-signal retention scoring,
	•	budget-aware transitions,
	•	exact mixed-tier attention,
	•	and replay-backed recovery for evicted content.

This document reports the path from the frozen design proposal to the final V1 freeze state, including the main implementation stages, benchmark data, observed bottlenecks, the optimization sequence that resolved them, and the final supported scope.

⸻

1. Introduction

KV-cache management is one of the central systems problems in long-context autoregressive inference. In decoder-only transformers, previously processed tokens remain available to future tokens through cached keys and values. This improves efficiency relative to recomputing the entire prefix, but also creates a memory-growth problem: historical KV state tends to grow monotonically with context length.

The standard solutions each have fundamental drawbacks.

Append-only full retention preserves all available historical information but gives memory the worst possible growth profile. Fixed sliding windows bound memory, but impose a coarse recency prior that can discard semantically important old context. Static quantization or static retention heuristics lower cost, but remain non-adaptive.

The present work targets a narrower but practically important systems question:

Can one build a semantically safe adaptive KV-cache residency policy that improves memory behavior over append-only retention and fixed windows without changing transformer attention semantics?

The answer pursued here is yes, but only under a strict design discipline:
	•	no training,
	•	no semantic modification of attention,
	•	no hidden approximation of logits,
	•	explicit replay after hard eviction,
	•	and a production-oriented implementation inside a real inference system.

⸻

2. Problem Statement

2.1 Baseline limitations

For a conversation of growing length, the ordinary KV-cache footprint grows approximately linearly with the number of retained tokens. In real systems, this leads to one of two unsatisfactory outcomes:
	1.	memory grows until it becomes a hard constraint, or
	2.	history is truncated by a static policy that ignores observed usefulness.

This is especially problematic in multi-turn settings with topic drift. Older context can become irrelevant for long intervals, yet remain resident. Conversely, simple recency-based trimming can remove old but still governing constraints.

2.2 Desired system properties

The target system must satisfy all of the following:
	•	preserve transformer semantics,
	•	observe historical usefulness externally,
	•	gradually demote cold regions before final eviction,
	•	support exact recovery after eviction via replay,
	•	operate with bounded control complexity,
	•	and remain implementable inside a production-grade inference engine.

2.3 Scope of the final system

The final frozen scope is intentionally restricted to keep the system rigorous and production-serious:
	•	single-request generation path only,
	•	model_type == "llama",
	•	verified full-attention Llama baseline only,
	•	homogeneous list[KVCache] model cache layout,
	•	block-level policy,
	•	FULL / COMPRESSED / EVICTED tiers,
	•	real replay-backed recovery,
	•	exact mixed-tier attention,
	•	and explicit rejection of unsupported families and modes.

⸻

3. Proposed Approach

3.1 Core design principle

The frozen design begins from a strict separation:
	•	attention semantics remain unchanged,
	•	residency policy is external.

The system therefore does not alter:
	•	Q,
	•	K,
	•	V,
	•	attention logits,
	•	or final attention outputs.

Instead, it observes usage and controls what resident KV state remains available in each residency tier.

3.2 Block-level memory model

The minimum policy unit is a contiguous block of tokens. Blocks are ordered chronologically and tracked through a logical registry. The residency policy acts on blocks, not individual tokens.

This choice reduces:
	•	metadata overhead,
	•	control-path fragmentation,
	•	locality loss,
	•	and implementation complexity.

3.3 Tier model

Each logical block belongs to one of three residency states:
	•	FULL: full-fidelity resident KV,
	•	COMPRESSED: lower-cost resident representation,
	•	EVICTED: no resident KV; only metadata and source-span references remain.

The ordinary transition flow is:

FULL -> COMPRESSED -> EVICTED

Hard eviction is real. Recovery requires replay.

3.4 Retention score

Let each block b maintain a bounded retention state derived from observed usage. The score is not a semantic quantity; it is a residency utility estimate.

A practical score decomposition used by the design is:

S_b = w_h H_b + w_p P_b + w_s R_b - w_a A_b

where:
	•	H_b is recent hotness,
	•	P_b is long-timescale persistence,
	•	R_b is a bounded structural prior,
	•	A_b is age/staleness,
	•	and w_h, w_p, w_s, w_a are bounded policy weights.

The recurrence is implemented with smooth decay and recovery rather than binary hit/miss rules. Conceptually:

H_b^{(t+1)} = \rho_h H_b^{(t)} + (1-\rho_h) U_b^{(t)}

P_b^{(t+1)} = \rho_p P_b^{(t)} + (1-\rho_p) U_b^{(t)}

A_b^{(t+1)} = A_b^{(t)} + \lambda_a \cdot \mathbf{1}[U_b^{(t)} = 0]

with:
	•	U_b^{(t)}: observed usage over a policy window,
	•	\rho_h, \rho_p: decay parameters,
	•	\lambda_a: staleness growth rate.

This family of updates preserves:
	•	graduality,
	•	boundedness,
	•	recoverability before eviction,
	•	and interpretability.

3.5 Usage observation

Usage is derived from attention participation after normal semantic computation, never by modifying attention behavior. The final implementation preserves observer-only semantics while gathering block-level usage summaries over policy windows.

3.6 Budget-aware control

The policy uses two memory thresholds:
	•	soft budget, where demotion becomes more aggressive,
	•	hard budget, where pressure must actively move resident memory downward unless only protected blocks remain.

Thus, the final policy is not score-only; it is score-guided but budget-aware.

3.7 Exact recovery

If an evicted block becomes necessary again, it is replayed from authoritative source-token spans. This preserves semantic correctness at the cost of explicit recovery work.

⸻

4. Mathematical / Systems Model

4.1 Resident state

For a retained prefix of blocks B = \{b_1, \dots, b_n\}, the resident state at time t is the ordered concatenation of all non-evicted block representations:

\mathcal{R}^{(t)} = \operatorname{concat}(r_{b_i}^{(t)} \mid b_i \notin E)

where r_{b_i}^{(t)} is either:
	•	full-fidelity KV state,
	•	compressed resident state,
	•	or absent if evicted.

4.2 Exact mixed-tier attention

For a current query tensor Q, the final implementation computes attention over ordered resident segments. If resident state is split into segments \{(K_j, V_j)\}_{j=1}^m, then scores are computed per segment and concatenated along the key axis:

Z_j = QK_j^T

Z = \operatorname{concat}(Z_1, \dots, Z_m)

A single global mask and global softmax are then applied:

\alpha = \operatorname{softmax}(Z + M)

The output is computed exactly as the sum of segment contributions under the global weights. This preserves semantic equivalence to attention over the fully assembled resident state, but may incur higher systems cost if the segment structure is fragmented.

4.3 Recovery condition

If a required historical prefix references a block b in state EVICTED, then before the next forward:

\text{recover}(b) := \text{replay from source span of } b

and the reconstructed state is reinserted as resident compressed state.

⸻

5. Implementation 1: Initial Integrated Version

5.1 First integrated shape

The first integrated MLXs version introduced:
	•	a new adaptive_kv subsystem,
	•	block registry,
	•	usage collector,
	•	score engine,
	•	transitions,
	•	eviction logic,
	•	ghost metadata,
	•	recovery interfaces,
	•	and compatibility gating.

The implementation used MLXs generation orchestration as the main control boundary:
	•	generate()
	•	chunked_prefill()
	•	decode_loop()

5.2 Early benchmarkable but not production-safe state

This version was architecturally coherent but still flawed in ways only serious audits exposed.

⸻

6. First Audit and Critical Problems

A first deep audit concluded the system was not ready.

6.1 Problem: semantic corruption after eviction

Blocks could be evicted during active generation and decode could proceed without replay restoring required context first.

This violated the strongest semantic requirement of the system.

6.2 Problem: fake resurrection

An EVICTED block could re-enter a higher logical tier without genuine resident reconstruction, breaking the invariant that logical span and resident state must remain aligned.

6.3 Problem: incorrect budgeting

Budget control used slab capacity rather than live resident bytes, making pressure signals inaccurate.

6.4 Problem: hard-pressure control was not truly hard-budget-driven

Transition guards could keep the system above budget even when actionable demotions existed.

6.5 Problem: compatibility was too broad

Unsupported llama subcases were admitted.

6.6 Problem: hot path was too expensive

The initial mixed-tier path suffered from heavy resident reassembly and observer overhead.

⸻

7. Implementation 2: Semantic Hardening

The next sequence of passes transformed the implementation into a production-serious V1 core.

7.1 Recovery barrier and replay-backed rematerialization

The system gained a pre-forward residency barrier. Before a forward that would otherwise consume missing history, evicted blocks are recovered through replay from source-token spans.

This closed the semantic corruption hole.

7.2 Hard eviction semantics finalized

After the fix:
	•	EVICTED blocks cannot return via metadata-only promotion,
	•	replay is mandatory for re-entry,
	•	and resident correctness is enforced at the manager layer.

7.3 Live-byte budgeting

Budgeting was changed to use actual live resident bytes rather than backing capacity.

7.4 Hard-pressure correctness

Hard pressure was corrected so non-protected blocks could not remain undemoted merely because of dwell/cooldown.

7.5 Compatibility narrowing

Support was limited to the exact safe baseline.

⸻

8. Benchmarks After Semantic Hardening

Once semantic correctness was repaired, main benchmark campaigns could be trusted.

8.1 Early benchmark outcome

Representative early results showed:
	•	correctness preserved,
	•	adaptive_full only modestly slower than non_adaptive,
	•	but adaptive_soft / adaptive_hard substantially slower.

At this stage, the compressed path was the main performance concern.

8.2 Initial interpretation

The first interpretation was that segmented mixed-tier attention cost was dominated by fragmentation.

⸻

9. Implementation 3: Compressed-Run Coalescing

9.1 Motivation

Microbenchmarks showed severe slowdown as compressed-segment count increased. Adjacent compressed blocks were each being treated as independent attention segments.

9.2 Change

Adjacent compressed blocks were merged into persistent compressed runs.

This preserved:
	•	logical block metadata,
	•	block-level policy,
	•	exact mixed-tier attention,
	•	and block-level usage attribution,

while reducing:
	•	compressed segment count,
	•	quantized matmul launches,
	•	segment-loop overhead.

9.3 Benchmark ambiguity and stability study

An initial before/after comparison was too noisy to interpret. A later repeated-run stability study demonstrated that one-shot throughput comparisons were unreliable.

The updated microbenchmark then proved that compressed-run coalescing was structurally active.

However, performance remained far below the all-FULL path. So fragmentation was no longer the dominant unresolved bottleneck.

⸻

10. Discovery of the Real Bottleneck

A profiler-style study on the coalesced compressed-run case isolated the next problem.

10.1 Measured result

For representative mixed-tier runs, the dominant wall-time contributor was not the score engine itself but the observer path:
	•	record_usage_from_attention
	•	mx.eval(usage_by_token)
	•	and host-side .item() extraction

Together, this path could account for the overwhelming majority of wall time.

10.2 Interpretation

At this point the main systems bottleneck was:
	•	per-forward usage materialization,
	•	host synchronization,
	•	and scalar extraction.

The fix therefore needed to preserve the observer signal while moving its materialization cost away from the hot path.

⸻

11. Implementation 4: Deferred Usage Extraction

11.1 Old behavior

Every mixed-tier forward:
	•	reduced attention weights,
	•	forced device synchronization,
	•	and extracted per-block scalars on the host.

11.2 New behavior

The optimized implementation now:
	•	computes the same per-block observer signal on device,
	•	stacks those values into deferred device batches,
	•	schedules them asynchronously,
	•	and materializes them only at the policy-window flush.

11.3 Why this preserves semantics

The signal is unchanged:
	•	same source weights,
	•	same per-block token slices,
	•	same window aggregation,
	•	same score-update boundary.

Only the synchronization point moved.

11.4 Measured result

Representative profiler rerun:
	•	previous mixed-tier representative case: ~90.3 tok/s, usage extraction ~72.5% of wall
	•	after deferred usage extraction: ~196.9 tok/s, usage flush ~1.13% of wall
	•	all-FULL reference in same rerun: ~211.0 tok/s

This showed a decisive optimization win.

⸻

12. Final Main Benchmark Rerun

A final repeated benchmark rerun established the final V1 performance position.

12.1 SOFT regime

Across repeated medians on:
	•	long_static
	•	topic_drift
	•	delayed_topic_return
	•	hard_pressure_context

adaptive_soft tracked adaptive_full extremely closely.

Representative ratios:
	•	0.98
	•	0.97
	•	1.02
	•	0.98

This is the key result for the accepted V1 operating regime.

12.2 HARD regime

The first repeated HARD reruns showed that adaptive_hard improved strongly relative to the earliest pathological state, but still left a large gap versus adaptive_full under longer stressed contexts.

Representative ratios from that stage were:
	•	roughly 0.69–0.72 of adaptive_full on the 512-token HARD matrix,
	•	and roughly 0.37 of adaptive_full on the long 1024-token stress case before the dedicated HARD completion passes.

This established that HARD correctness was sound, but that the remaining issue had shifted from obvious semantic or control-path defects to a harder systems bottleneck.

12.3 Dedicated HARD completion track

The dedicated HARD workstream then proceeded through a narrow sequence of exact optimizations:
	•	HARD stabilization,
	•	replay-prefix shortening,
	•	scratch replay reuse,
	•	recovery-wave reduction,
	•	post-wave materialization optimization,
	•	replay-forward optimization,
	•	and residual executor investigation.

These passes did not broaden scope or weaken semantics. Instead they reduced futile eviction/recovery churn, tightened replay cost, and clarified what cost remained structural.

A later Exact Runtime Optimization Program then revisited the residual mixed-tier executor and resident/executor handoff with broader implementation latitude, while still keeping exactness, scope, and user-visible behavior fixed.

12.4 Final HARD position

By the retained late-stage HARD artifacts, the stressed regime had improved materially from the earlier long-context state.

Representative retained HARD references show:
	•	at 512 prompt tokens, adaptive_hard at 207.57 tok/s versus 234.18 tok/s for adaptive_full, with one eviction and one recompute request,
	•	at 1024 prompt tokens, adaptive_hard at 222.08 tok/s versus 307.94 tok/s for adaptive_full, with six evictions and one recompute request,
	•	and reference_token_match remaining true.

The remaining gap is therefore no longer best described as a replay-wave explosion, an observer-path bug, or an obvious manager-side inefficiency. It is structural to the current exact mixed-tier segmented executor under compressed-heavy HARD steady state.

The earlier retained runtime work did not keep every attempted change. A broader executor-fusion variant was benchmark-negative and was reverted. The retained subset kept executor-ready resident slice metadata, a score-side decode specialization for q_len == 1 mixed-tier resident state, and the generic segmented executor as the exact oracle and fallback.

On fresh same-machine before/after benchmark gating, that retained subset improved adaptive_soft by about 1.6% on long_static and 0.6% on hard_pressure_context, improved adaptive_hard by about 13.6% at 512 prompt tokens and 3.7% at 1024 prompt tokens, and left recompute counts, eviction counts, final pressure state, final resident bytes, and correctness signals semantically consistent.

A deeper structural continuation of the Exact Runtime Optimization Program then explored a more aggressive exact mixed-tier executor redesign. That redesign remained exact and passed validation, but the decisive same-process 1024-token HARD gate improved by only about 1.6%, below the intended retention threshold for materially more complex executor work. It was therefore reverted, and the retained runtime baseline remained the earlier implementation already described above.

12.5 Regression correctness and token parity

On the synthetic benchmark matrix used for V1 regression, reference_token_match remained true for adaptive_full versus non_adaptive in the final repeated rerun.

Real-workload checks later isolated two patterns that differ in meaning:

First, adaptive_full greedy mismatch on a short prompt workload (C5) was traced to usage sampling: the implementation temporarily took the main attention output off the fused scaled-dot-product path even when resident state was all-FULL and contiguous. The resolution keeps fused attention for the forward output in that configuration and derives usage weights from a separate full-precision scores path when sampling is required. After this change, adaptive_full is again expected to match non_adaptive token-for-token under greedy decoding whenever compression does not alter KV fidelity.

Second, adaptive_soft mismatch on a long markdown workload (T4) occurred only while compressed resident KV was active; adaptive_full continued to match the reference. A shadow dequantized resident check reproduced the same token flip, supporting classification as compression-induced numerical drift rather than a defect in mixed-tier control or policy wiring. V1 does not promise greedy token parity for adaptive_soft when COMPRESSED blocks participate.

12.6 HARD regime beyond throughput

adaptive_hard remains semantically safe and functional under stress, and after the dedicated HARD completion track it should be regarded as good enough within the approved V1 scope. The remaining performance gap is structural to the current mixed-tier segmented executor design, not a sign of incorrect residency logic or an unfinished ordinary hardening pass.

⸻

13. Discussion

13.1 What the final data mean

The final data support three strong conclusions.

First, the design is semantically viable. Exact replay-backed recovery, live-byte budgeting, and observer-only attention preservation are all compatible in a real inference system.

Second, the main target regime — SOFT compression without replay-heavy stress — is now performance-acceptable. The final implementation behaves much closer to adaptive_full than the earlier benchmark campaigns suggested.

Third, the remaining cost center has shifted. HARD-pressure behavior still has meaningful performance overhead, but after the completed HARD track that overhead is now best understood as structural to the exact segmented executor. It is no longer a blocker for accepting the V1 freeze within the approved scope.

13.2 Why HARD is not a blocker for V1

The project did not define success as universal parity across all pressure regimes. The actual approved V1 target was a semantically safe adaptive policy with credible performance in its supported regime.

That objective has been met. HARD is therefore closed for V1: good enough under stress within the approved scope, with any further gain requiring a different class of work than the hardening passes already completed.

13.3 What this work does not claim

This work does not claim:
	•	universal model-family support,
	•	batch/speculative compatibility,
	•	prompt-cache integration,
	•	full HARD-pressure throughput parity with adaptive_full,
	•	or greedy token-by-token parity for adaptive_soft when compressed resident KV is on the attention path, unless a separate compressed-KV fidelity program is undertaken.

The last two items are explicit non-goals or quality tradeoffs for V1, not failures of the adaptive policy core. Further HARD improvement now belongs to optional executor R&D rather than ordinary V1 completion work.

⸻

14. Final Freeze Position

Adaptive KV V1 should be considered:

good within scope

and frozen as:
	•	semantically correct within the approved scope, including replay-backed recovery and correct resident visibility,
	•	adaptive_full aligned with non_adaptive greedy outputs when resident KV is all-FULL contiguous and the fused attention path matches the baseline,
	•	adaptive_soft performant in the SOFT regime relative to adaptive_full, without a guarantee of greedy token parity under active compression,
	•	adaptive_hard materially improved and good enough under stress within the approved scope, while not held to adaptive_full throughput or parity,
	•	benchmarked,
	•	production-serious within scope,
	•	acceptable in performance for the supported SOFT compression regime,
	•	and bounded in remaining HARD limitations by the current exact mixed-tier segmented executor design.

The supported scope remains exactly the restricted matrix already defined earlier in this document.

Optional diagnostic note: a shadow dequantized resident experiment remains a lightweight way to separate quantization drift from adaptive control issues in future investigations.

⸻

15. Conclusion

This work started from a general systems problem: transformer KV memory is too expensive to treat as uniformly valuable over long multi-turn histories. The final answer was not to alter attention semantics, but to build an external observer-driven residency policy with exact recovery.

The project progressed through multiple serious engineering phases:
	•	frozen design,
	•	MLXs integration,
	•	semantic audits,
	•	replay-backed hardening,
	•	compressed-run optimization,
	•	usage-path optimization,
	•	dedicated HARD stabilization and replay optimization,
	•	exact runtime optimization with partial retention,
	•	residual executor assessment,
	•	repeated benchmarking,
	•	and final freeze.

The key result is not merely that adaptive KV can be made to work, but that it can be made:
	•	semantically safe,
	•	production-serious,
	•	and performant enough within a well-defined supported scope.

That is the correct stopping point for V1. The HARD track and the later Exact Runtime Optimization Program are closed, with the existing retained runtime baseline left in place; any further improvement belongs to optional deeper executor R&D only if product goals justify it.

⸻

16. Future Work

Future work should be pursued only if driven by real requirements.

16.1 Compressed-KV fidelity

If greedy parity or tighter output distributions are required while COMPRESSED blocks are resident, the natural lever is the quantization path itself (kernel choice, bit width, dequant and rounding alignment), not a rewrite of adaptive policy control. Shadow dequantized checks remain a practical triage tool.

16.2 HARD-pressure optimization

The ordinary HARD hardening track and the later Exact Runtime Optimization Program are complete for V1. The existing retained runtime baseline stays in place, and the remaining limitation is still structural to the current exact mixed-tier segmented executor.

Any further work should therefore be treated as optional executor R&D, for example:
	•	a deeper exact mixed-tier executor restructuring for the stable HARD decode shape,
	•	or narrower replay/executor studies tied to explicit product requirements.

16.3 Broader validation

More real prompts and broader repeated campaigns could strengthen confidence beyond synthetic scenarios.

16.4 Scope expansion

Any expansion to:
	•	more llama variants,
	•	other families,
	•	batch/speculative,
	•	prompt-cache interop,

must be treated as a new explicit design/integration track.