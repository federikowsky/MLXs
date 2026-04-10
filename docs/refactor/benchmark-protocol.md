MLXs — Benchmark Protocol

Status: canonical benchmark specification
Role: governs measurement, comparison, and validation of MLXs during the first refactor cycle
Upstream inputs:
	•	MLXs — Vision / Architectural Foundations
	•	MLXs Research Dossier
	•	MLXs — Performance Core Spec

This document defines what it means to benchmark MLXs correctly during the first refactor cycle. It does not redefine the core contract or implementation strategy.

⸻

1. Purpose of the Benchmark Protocol

This protocol exists to make benchmark results:
	•	fair against the current mlx-lm baseline,
	•	reproducible across runs and revisions,
	•	aligned with the layered architecture of the refactor,
	•	interpretable in terms of the Performance Core rather than mixed product behavior.

It governs:
	•	benchmark taxonomy,
	•	workload isolation,
	•	fairness rules,
	•	measurement boundaries,
	•	metric definitions,
	•	reporting policy,
	•	pass/fail/regression criteria.

It is necessary because MLX’s lazy execution model makes benchmark boundaries architecturally meaningful: graph construction, explicit eval(), implicit evaluation through item(), and host-side conversions can all change what is actually being measured.  ￼

⸻

2. Benchmarking principles

2.1 Fairness

A benchmark is fair only if MLXs and mlx-lm are measured on the same workload class with materially equivalent generation settings and comparable execution conditions. The mlx-lm baseline is not abstract: its comparable generation path currently uses a dedicated generation stream, builds work under with mx.stream(generation_stream), issues mx.async_eval(y, logprobs) during decode, materializes the first generated token with mx.eval(y), yields y.item(), logprobs, and calls mx.clear_cache() every 256 decode iterations.  ￼

2.2 Reproducibility

A benchmark must document:
	•	hardware,
	•	OS,
	•	MLX version,
	•	mlx-lm version or commit,
	•	MLXs revision,
	•	model and quantization,
	•	tokenizer equivalence,
	•	warmup policy,
	•	number of timed runs,
	•	reporting aggregation,
	•	whether compile is enabled,
	•	whether cache-clearing policy is normalized.

Because MLX documents fixed overhead per graph evaluation and graph-size-dependent overhead, benchmark boundaries must be stable and documented, not inferred after the fact.  ￼

2.3 Workload isolation

Fast-path benchmarking must be isolated from:
	•	richer sampling semantics,
	•	penalties/processors,
	•	speculative decoding,
	•	prompt-cache reuse,
	•	server framing and transport,
	•	broader product surfaces.

This follows from the layered architecture and from the fact that unused extra outputs still incur graph-build cost in MLX.  ￼

2.4 Architectural relevance

Each benchmark class must map to a specific architectural surface:
	•	Performance Core,
	•	Generation Semantics,
	•	Advanced Engines,
	•	Product Surfaces.

A benchmark is invalid if it mixes layers without declaring that mixed scope.

2.5 Separation between fast-path and feature-breadth benchmarks

Fast-path benchmarking governs the first implementation cycle. Broader features are benchmarked separately so that feature breadth does not redefine the initial notion of success.

⸻

3. Benchmark classes

The canonical benchmark classes are:

3.1 Class A — Minimal fast-path decode

Purpose:
	•	govern the first implementation cycle,
	•	measure the narrowest competitive path,
	•	validate the Performance Core.

Scope:
	•	single request,
	•	text-only,
	•	same model,
	•	same tokenizer-equivalent input,
	•	plain generation cache,
	•	no speculative,
	•	no prompt-cache reuse,
	•	no server transport overhead,
	•	no broader product shaping.

This is the governing class for the first MLXs refactor cycle.

3.2 Class B — Enriched single-request generation

Purpose:
	•	measure generation semantics above the core.

Scope may include:
	•	richer sampling,
	•	penalties/processors,
	•	logprobs/top-logprobs,
	•	richer finish behavior.

This class does not govern the initial fast-path decision.

3.3 Class C — Speculative decoding

Purpose:
	•	measure speculative runtime surfaces as a separate Advanced Engines benchmark class.

Scope:
	•	draft model configuration,
	•	acceptance/rejection behavior,
	•	throughput gains or regressions versus non-speculative baselines.

3.4 Class D — Prompt-cache reuse

Purpose:
	•	measure prompt-cache reuse as a separate class rather than polluting minimal decode measurement.

Scope:
	•	same-prefix reuse,
	•	reuse hit/miss conditions,
	•	cache state reuse costs and gains.

3.5 Class E — Server / streaming path

Purpose:
	•	measure product-surface behavior, including transport and streaming response behavior.

Scope:
	•	HTTP/SSE or equivalent server path,
	•	server-side framing,
	•	end-to-end latency,
	•	streaming overhead,
	•	request handling overhead.

3.6 Class F — Memory and execution diagnostics

Purpose:
	•	collect diagnostic measurements outside the primary success metric path.

Scope:
	•	graph export,
	•	synchronization proxies,
	•	materialization proxies,
	•	compile warmup cost,
	•	memory-pressure proxies where available.

This class is supportive, not the primary success criterion.

⸻

4. Canonical benchmark for MLXs v1 core

The canonical benchmark for the first implementation cycle is:

Class A — Minimal fast-path decode

It governs the first cycle because it is:
	•	narrow,
	•	comparable,
	•	architecturally aligned with the Performance Core,
	•	measurable without mixing in broader semantics.

Canonical workload definition

The benchmark must measure a single-request decode workload with:
	•	the same model on both MLXs and mlx-lm,
	•	the same quantization/layout class where applicable,
	•	token-equivalent prompt input,
	•	identical prompt length target,
	•	identical max generated token target,
	•	the same minimal generation regime,
	•	no server transport,
	•	no speculative decoding,
	•	no prompt-cache reuse benchmark semantics,
	•	no product-surface event shaping in the primary timed region.

Purpose

This benchmark governs:
	•	whether the Performance Core is competitive,
	•	whether the refactor is moving in the right direction,
	•	whether later benchmark classes should proceed.

This benchmark does not govern broader feature parity.

⸻

5. Fairness rules versus mlx-lm

5.1 Same model

MLXs and mlx-lm must use the same model artifact or functionally identical model artifact.

5.2 Same quantization / weight class

If one side uses a quantized model or a different weight format with different runtime implications, that must be normalized or explicitly declared as a different benchmark case.

5.3 Same tokenization or token-equivalent input

Comparison must use either:
	•	identical tokenizer and tokenized input,
	•	or a token-equivalent pre-tokenized input stream.

No benchmark may compare differently tokenized prompt content and claim decode fairness.

5.4 Same generation regime

For Class A, generation settings must be aligned to a minimal, comparable regime. mlx-lm’s server surface exposes temperature, top_p, top_k, min_p, repetition/presence/frequency penalties, logprobs, and speculative controls, but these belong to broader benchmark classes unless explicitly included.  ￼

5.5 Same prompt length and decode target

Prompt token target and generation token target must be identical.

5.6 Same workload class

Minimal decode must be compared with minimal decode, not with enriched server-generation surfaces.

5.7 Explicit handling of known mlx-lm runtime choices

The protocol must explicitly account for known mlx-lm decode-path behavior:
	•	dedicated generation stream,
	•	mx.async_eval(y, logprobs) in decode,
	•	mx.eval(y) for the first generated token,
	•	yield y.item(), logprobs,
	•	periodic mx.clear_cache() at n % 256 == 0.  ￼

Fairness requires that MLXs either:
	•	normalize these choices where materially relevant,
	•	or explicitly document the divergence and treat it as part of the comparison contract.

⸻

6. Benchmark setup

6.1 Hardware target

Primary benchmark target is Apple Silicon hardware running MLX.

The exact benchmark report must declare:
	•	machine class,
	•	SoC,
	•	RAM,
	•	macOS version.

6.2 Software assumptions

The benchmark report must declare:
	•	MLX version,
	•	mlx-lm commit/version,
	•	MLXs revision,
	•	Python version,
	•	relevant dependency versions if they affect runtime behavior.

6.3 Runtime assumptions

The benchmark must declare:
	•	eager or compiled path,
	•	stream policy under test,
	•	cache-clearing policy,
	•	whether diagnostics are disabled,
	•	whether graph export is disabled in the primary measurement run.

6.4 Cold-start vs warm-start

Cold-start and warm-start must be separated.

Rules:
	•	primary fast-path throughput claims are warm-state claims unless explicitly labeled cold-start;
	•	compile warmup, if applicable, must not be merged invisibly into steady-state decode numbers;
	•	initial model load and compile cost must be separately reportable.

6.5 Warmup policy

Each benchmark class must define warmup explicitly.

For the canonical Class A benchmark:
	•	warmup runs are required before timed runs,
	•	warmup count must be fixed and reported,
	•	the same warmup policy must apply to both MLXs and mlx-lm,
	•	compile warmup must be explicit if compile is enabled.

6.6 Number of runs

Timed runs must be multiple, not single-shot.

Reporting must include:
	•	median,
	•	variance-aware context such as min/max or dispersion,
	•	optionally first-run versus later-run reporting if session effects are relevant.

6.7 Reporting policy

The canonical report must include:
	•	prompt token target,
	•	decode token target,
	•	run count,
	•	aggregation rule,
	•	warmup rule,
	•	hardware/software declaration,
	•	benchmark class label.

6.8 Variance handling

If variance is high, the benchmark must not silently collapse everything into one scalar. It must:
	•	expose first-run and aggregated results where architecturally relevant,
	•	separate session-stability concerns from steady-state decode throughput,
	•	document any thermal or runtime-state caveats observed during the session.

⸻

7. Metrics

7.1 TTFT

Definition: time from benchmarked request start to first generated token becoming available at the chosen measurement boundary.

TTFT matters in:
	•	Class A when seed-step cost is architecturally relevant,
	•	Class E for user-facing serving behavior.

7.2 Steady-state decode tok/s

Definition: generated tokens per second over the decode region after initial token availability, under the declared measurement boundary.

This is the primary metric for Class A.

7.3 End-to-end latency

Definition: total latency across the full measured workload boundary.

Relevant for:
	•	Class B,
	•	Class E,
	•	selected Class A summaries.

It is not the primary success metric for the initial fast-path benchmark.

7.4 Peak memory

Where available, the benchmark must capture peak memory or a reproducible proxy.

Relevant for:
	•	Class A summary reporting,
	•	Class D diagnostics,
	•	regression detection.

7.5 Temporary allocation pressure or proxy

If direct allocation-pressure metrics are unavailable, the protocol allows:
	•	documented memory proxies,
	•	runtime-side counters,
	•	repeated-run memory tracking,
provided they are labeled as proxies, not direct allocator truth.

7.6 Eval / materialization / sync-related measurements

The protocol should collect observable proxies for:
	•	explicit eval() count or placement,
	•	scalar extraction count,
	•	materialization count,
	•	synchronization count or declared synchronization boundaries.

These are diagnostic metrics, not primary victory metrics.

7.7 Compile warmup cost

If compile is enabled in a benchmark class, compile warmup cost must be separately reported.

It must never be silently blended into steady-state decode throughput claims.

⸻

8. Inclusion / exclusion rules

8.1 Included in the canonical fast-path benchmark

The canonical Class A benchmark includes only:
	•	prompt ingestion required for the benchmarked request,
	•	active generation runtime,
	•	active cache usage required by that request,
	•	minimal token-selection behavior required by the chosen minimal generation regime,
	•	declared synchronization/evaluation behavior inside the runtime boundary.

8.2 Excluded from the canonical fast-path benchmark

The canonical Class A benchmark excludes:
	•	server transport,
	•	SSE framing,
	•	product-layer event shaping,
	•	speculative decoding,
	•	prompt-cache reuse benchmarking,
	•	richer penalties/processors,
	•	broad logprobs behavior,
	•	rich observability instrumentation,
	•	graph-export work,
	•	diagnostics that alter runtime shape.

8.3 Features that require separate benchmarks

Separate benchmark classes are required for:
	•	logprobs/top-logprobs,
	•	penalties/processors,
	•	speculative decoding,
	•	prompt-cache reuse,
	•	server/streaming path,
	•	compile warmup behavior,
	•	graph diagnostics and shape inspection.

⸻

9. Measurement boundaries

9.1 What is measured

Each benchmark class must declare exactly what region is timed.

For Class A, the report must separately identify:
	•	prefill region,
	•	decode region,
	•	TTFT boundary,
	•	full end-to-end region if also reported.

9.2 Start and end boundaries

The benchmark must define:
	•	when timing starts,
	•	when timing stops,
	•	whether timing begins before or after warmup,
	•	whether timing includes prompt ingestion only, decode only, or both.

9.3 Treatment of prefill

Prefill must be measurable separately from decode.

Reason:
	•	prefill and decode are architecturally distinct in the Performance Core,
	•	MLX boundary behavior can differ materially between prompt processing and steady decode.

9.4 Treatment of decode

Decode throughput must be measured over a declared decode region, not estimated from end-to-end time unless the benchmark class explicitly says so.

9.5 Treatment of sampling

Sampling must be included only when the benchmark class includes it by definition.

For Class A, the generation regime should stay minimal and comparable.

9.6 Treatment of finish conditions

Finish behavior must be normalized:
	•	fixed max-tokens benchmark is preferred for primary comparability,
	•	stop-sequence or EOS-driven variation must be separately declared if used.

9.7 Observability and diagnostics exclusion

Instrumentation that materially alters runtime behavior must be excluded from the primary measurement run.

That includes:
	•	graph export,
	•	heavy profiling wrappers,
	•	debug printing of arrays,
	•	conversions that force evaluation.

Because MLX treats printing and NumPy conversion as evaluation triggers, even diagnostic code can alter benchmark behavior.  ￼

⸻

10. Instrumentation guidance

10.1 Signals and counters to collect

The benchmark harness should collect:
	•	TTFT,
	•	decode tok/s,
	•	end-to-end latency where relevant,
	•	peak memory or proxy,
	•	warmup count,
	•	timed run count,
	•	prompt length,
	•	decode length,
	•	compile on/off state,
	•	stream policy label,
	•	cache-clearing policy label,
	•	materialization/eval/sync proxy counts where available.

10.2 Observing sync, eval, scalar extraction, and materialization

Primary benchmark runs should use low-perturbation counters or declared boundary labels instead of invasive instrumentation.

Rules:
	•	do not insert extra eval() just to count it,
	•	do not insert debug scalar reads in hot loops,
	•	do not use graph export in the timed path,
	•	if proxy counters require behavioral perturbation, collect them in a separate diagnostics run.

10.3 Graph export and related diagnostics

Graph export should be used only outside the primary timing run.

Reason:
	•	it is a diagnostics tool for shape/cost reasoning,
	•	it belongs to Class F,
	•	it must not contaminate the canonical fast-path measurement.

⸻

11. Pass / fail / regression criteria

11.1 Victory

A benchmark result is a victory for the target class if:
	•	MLXs is measurably faster on the primary class metric,
	•	no higher-priority metric regresses beyond tolerance,
	•	the result is stable across the declared reporting policy.

For Class A, the primary class metric is steady-state decode tok/s.

11.2 Parity

A result is parity if:
	•	MLXs and mlx-lm differ only within declared tolerance bands,
	•	no meaningful regression appears in TTFT or memory behavior for the same workload class.

11.3 Regression

A result is a regression if:
	•	the primary metric is materially worse than the baseline,
	•	or a higher-priority metric degrades enough to outweigh gains in a lower-priority metric.

11.4 Tolerances

Tolerance bands must be explicit in each benchmark report.

The protocol does not hardcode numeric tolerances here; they must be frozen consistently across benchmark runs once the first implementation cycle’s hardware/software baseline is locked.

11.5 Priority ordering in tradeoffs

If metrics conflict, the priority order is:
	1.	correctness of workload equivalence,
	2.	Class A primary metric for the governing benchmark,
	3.	TTFT for that benchmark class,
	4.	memory behavior,
	5.	lower-priority diagnostics.

For the first implementation cycle, Class A governs the fast path.

⸻

12. Architectural consequences

12.1 Class A — Minimal fast-path decode

Architectural consequence: This class governs the initial fast path and therefore governs the first implementation cycle of the Performance Core.

12.2 Class B — Enriched single-request generation

Architectural consequence: This class validates the Generation Semantics layer and must not be used to redefine the initial core success condition.

12.3 Class C — Speculative decoding

Architectural consequence: This class belongs to Advanced Engines and must remain separate from initial core victory criteria.

12.4 Class D — Prompt-cache reuse

Architectural consequence: Prompt-cache reuse is a separate architecture surface and must not pollute the minimal fast-path benchmark.

12.5 Class E — Server / streaming path

Architectural consequence: Product-surface benchmarking belongs after the core and general path are measurable independently.

12.6 Class F — Memory and execution diagnostics

Architectural consequence: Diagnostics must support design and debugging, but must remain outside primary success measurement.

⸻

13. Open questions

The following remain intentionally open:
	1.	Exact numeric tolerance bands for victory/parity/regression, pending locked hardware/software baseline for the first implementation cycle.
	2.	Exact canonical prompt/decode workload matrix for Class A, pending final first-cycle benchmark suite definition.
	3.	Exact normalization of cache-clearing policy versus mlx-lm, beyond the requirement that it be explicit and fair.
	4.	Exact compile benchmark treatment once a compile seam exists in MLXs.
	5.	Exact diagnostic proxy set for synchronization/materialization counts that does not materially perturb measurement.
	6.	Whether first-run and later-run reporting must both be mandatory in the canonical report for all classes or only for classes sensitive to session effects.

These are not implementation blockers for the protocol itself, but they must be resolved before benchmark execution is declared final.

⸻

This document is now the canonical reference for measuring MLXs correctly in the first refactor cycle.
The next correct artifact is the Repo Baseline Audit, because the architecture, core contract, and benchmark protocol are now defined well enough to decide what codebase baseline to retain, discard, or rebuild from.