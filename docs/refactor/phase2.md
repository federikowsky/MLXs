MLXs — Phase 2 Execution Checklist

Status: execution artifact
Role: operational checklist for Phase 2 of the MLXs refactor

⸻

1. Purpose of Phase 2

Phase 2 exists to validate the new Layer 1 / Performance Core against the canonical benchmark model before any Layer 2 reattachment begins.

It must make four things true:
	•	the benchmark harness is aligned with the canonical Benchmark Protocol;
	•	the benchmark is measuring the new Layer 1, not legacy runtime paths;
	•	the comparison against mlx-lm is fair, explicit, and reproducible;
	•	benchmark drift, contamination, and hidden measurement errors are excluded before richer semantics are reintroduced.

Phase 2 comes before Layer 2 reattachment because the canonical fast-path benchmark is the governing validation for the first refactor cycle. If Layer 2 is reattached first, benchmark interpretation becomes ambiguous and the canonical Class A benchmark can no longer be trusted as a pure Layer 1 validation.

⸻

2. Phase 2 goals
	•	Align the benchmark harness to the canonical Benchmark Protocol.
	•	Freeze the benchmarked workload as Class A — Minimal fast-path decode.
	•	Confirm that the benchmark path executes the new Layer 1 subtree.
	•	Confirm fairness against the current mlx-lm baseline.
	•	Confirm measurement boundaries are explicit and stable.
	•	Exclude contamination from Layer 2, Layer 3, Layer 4, diagnostics, and transitional shims.
	•	Produce reproducible benchmark outputs suitable for Phase 2 sign-off and later comparison.

⸻

3. Benchmark harness alignment checklist

3.1 Workload-class labeling
	•	Confirm the canonical benchmark is labeled explicitly as Class A — Minimal fast-path decode.
	•	Confirm that no Layer 2, Layer 3, or Layer 4 benchmark is mixed into the canonical Class A path.
	•	Confirm that any non-Class-A benchmark path is separately labeled and not used for Phase 2 sign-off.

3.2 Warmup rules
	•	Confirm warmup runs are explicit and fixed.
	•	Confirm the same warmup policy is applied to MLXs and mlx-lm.
	•	Confirm warmup is excluded from timed metrics.
	•	Confirm compile warmup, if any path supports compile later, is separately reportable and not blended into steady-state results.

3.3 Reporting rules
	•	Confirm benchmark reports include:
	•	workload class,
	•	hardware target,
	•	OS,
	•	MLX version,
	•	mlx-lm revision/version,
	•	MLXs revision,
	•	model identifier,
	•	quantization / weight-format class,
	•	warmup count,
	•	timed run count,
	•	aggregation rule.
	•	Confirm first-run and aggregated reporting policy is explicit if session sensitivity is relevant.
	•	Confirm no benchmark result is accepted without benchmark-class labeling.

3.4 Measurement boundary handling
	•	Confirm the harness distinguishes prefill, decode, TTFT, and end-to-end regions when applicable.
	•	Confirm timed regions start and stop at explicit declared boundaries.
	•	Confirm the harness does not infer boundaries implicitly from mixed runtime behavior.

⸻

4. Canonical Class A readiness checklist

The following must be true before canonical Layer 1 benchmarking begins.

4.1 Same model
	•	Confirm MLXs and mlx-lm use the same model artifact or a functionally identical model artifact.

4.2 Same quantization / weight class
	•	Confirm quantization or weight-format class is identical or explicitly normalized.
	•	Confirm no quantization mismatch is being treated as a fair decode benchmark.

4.3 Same tokenization or token-equivalent input
	•	Confirm the benchmark uses identical tokenizer output or token-equivalent prompt input on both sides.
	•	Confirm prompt equivalence is not assumed from raw text alone.

4.4 Same prompt/decode target
	•	Confirm identical prompt token target.
	•	Confirm identical decode token target / max token target.
	•	Confirm finish conditions are normalized for the benchmark class.

4.5 Same generation regime
	•	Confirm the benchmark uses the canonical minimal generation regime for Class A.
	•	Confirm no Layer 2 semantic enrichments are silently included.
	•	Confirm no Layer 3 orchestration behavior is silently included.
	•	Confirm no Layer 4 transport/product behavior is silently included.

⸻

5. Fairness versus mlx-lm checklist

5.1 Baseline path confirmation
	•	Confirm the mlx-lm path being compared is the current comparable generation path, not an unrelated product surface.
	•	Confirm the benchmark report records which mlx-lm path/surface is being compared.

5.2 Known runtime differences to check explicitly
	•	Confirm how stream usage is handled on both sides.
	•	Confirm how mx.async_eval is used on both sides.
	•	Confirm how mx.eval is used on both sides.
	•	Confirm where scalar extraction via item() or equivalent occurs on both sides.
	•	Confirm whether mx.clear_cache() is used and at what cadence on both sides.

5.3 Handling known mlx-lm behaviors

The current mlx-lm generation path is known to use:
	•	a generation stream,
	•	mx.async_eval(y, logprobs),
	•	mx.eval(y) for the first generated token,
	•	yield y.item(), logprobs,
	•	mx.clear_cache() every 256 decode iterations. (github.com￼)
	•	Confirm each of these is either normalized, matched, or explicitly documented as a deliberate divergence.
	•	Confirm any deliberate divergence is recorded as part of benchmark interpretation, not hidden.

5.4 Documentation of deliberate divergences
	•	Confirm every deliberate divergence from mlx-lm runtime behavior is documented in the benchmark report.
	•	Confirm no result is presented as “fair comparison” if meaningful divergences are undocumented.

⸻

6. Measurement boundary checklist

6.1 Prefill boundary
	•	Confirm prefill start is explicit.
	•	Confirm prefill end is explicit.
	•	Confirm prefill can be reported separately from decode.

6.2 Decode boundary
	•	Confirm decode region start is explicit.
	•	Confirm decode region end is explicit.
	•	Confirm steady-state decode tok/s is derived from the declared decode region, not inferred from mixed regions.

6.3 TTFT boundary
	•	Confirm TTFT start is explicit.
	•	Confirm TTFT end corresponds to first generated token availability at the declared boundary.
	•	Confirm TTFT is not conflated with full end-to-end latency.

6.4 End-to-end boundary
	•	Confirm end-to-end is reported only if explicitly declared.
	•	Confirm end-to-end does not replace decode metrics for Class A.

6.5 Included vs excluded work
	•	Confirm the benchmark report states what work is included in each metric.
	•	Confirm excluded work is also documented where necessary to interpret results.
	•	Confirm no hidden diagnostics or transport work is included in canonical Class A timing.

⸻

7. Benchmark contamination exclusion checklist

7.1 Layer 2 exclusion
	•	Confirm non-canonical Layer 2 rich semantics are not in the timed Class A path.
	•	Confirm logprobs/top-logprobs are excluded unless the canonical class explicitly includes them.
	•	Confirm penalties/processors are excluded from the canonical Class A path.

7.2 Layer 3 exclusion
	•	Confirm batching is excluded from the canonical Class A path.
	•	Confirm speculative decoding is excluded from the canonical Class A path.
	•	Confirm prompt-cache reuse benchmarking is excluded from the canonical Class A path.
	•	Confirm no Layer 3 scheduler/admission behavior is contaminating the canonical Layer 1 benchmark.

7.3 Layer 4 exclusion
	•	Confirm server transport is excluded.
	•	Confirm SSE/chunk framing is excluded.
	•	Confirm CLI/product-surface rendering is excluded.
	•	Confirm compatibility/API-shaping overhead is excluded.

7.4 Diagnostics exclusion
	•	Confirm graph export is not in the timed path.
	•	Confirm debug printing of arrays is absent from the timed path.
	•	Confirm diagnostic NumPy conversions are absent from the timed path.
	•	Confirm invasive profiling wrappers are excluded from primary timing runs.

7.5 Temporary shim exclusion
	•	Confirm no transitional compatibility shim is on the canonical benchmark path if it changes runtime behavior materially.
	•	Confirm any unavoidable shim on the benchmark path is explicitly documented and treated as a temporary benchmark caveat.

⸻

8. Instrumentation discipline checklist

8.1 What may be observed
	•	TTFT
	•	steady-state decode tok/s
	•	end-to-end latency where declared
	•	warmup count
	•	run count
	•	prompt/decode lengths
	•	compile on/off label
	•	stream policy label
	•	cache-clearing policy label
	•	peak memory or declared proxy
	•	declared eval/materialization/sync proxies where they do not perturb the timed path

8.2 What must stay outside the timed path
	•	graph export
	•	diagnostic scalar reads added only for instrumentation
	•	debug array printing
	•	diagnostic NumPy conversion
	•	invasive profiling that changes evaluation behavior
	•	product-surface logging/telemetry not part of the canonical benchmark class

8.3 Eval/materialization/sync proxies
	•	Confirm proxies for eval/materialization/sync are collected only if they do not alter the timed path materially.
	•	If such proxies would perturb the timed path, confirm they are collected in a separate diagnostics run.
	•	Confirm diagnostics runs are labeled as non-primary measurement runs.

⸻

9. Phase 2 validation checklist

9.1 Validate that the benchmark measures the new Layer 1
	•	Confirm the canonical benchmark path traverses the new Layer 1 subtree.
	•	Confirm the old universal generate/decode center is not being benchmarked by accident.
	•	Confirm any remaining compatibility layers are not redefining the measured runtime path.

9.2 Validate reproducibility
	•	Confirm the benchmark environment is recorded.
	•	Confirm the benchmark inputs are stable and repeatable.
	•	Confirm warmup and timed-run counts are fixed and repeatable.
	•	Confirm aggregation/reporting rules are stable and repeatable.

9.3 Validate Benchmark Protocol conformance
	•	Confirm workload-class separation matches the Benchmark Protocol.
	•	Confirm fairness rules are satisfied or explicitly documented when diverging.
	•	Confirm measurement boundaries match the protocol.
	•	Confirm contamination exclusions are enforced.
	•	Confirm instrumentation discipline is respected.

9.4 Validate result interpretability
	•	Confirm each reported metric maps to a declared boundary.
	•	Confirm each reported result names the workload class.
	•	Confirm benchmark caveats are explicit.
	•	Confirm no mixed-layer result is presented as a pure Layer 1 benchmark.

⸻

10. Phase 2 exit criteria

Phase 2 is complete only when all of the following are true:
	•	The benchmark harness is aligned to the canonical Benchmark Protocol.
	•	The canonical Class A benchmark is runnable and reproducible.
	•	The benchmark path measures the new Layer 1 subtree.
	•	Fairness versus mlx-lm is explicitly checked and documented.
	•	Known runtime differences versus mlx-lm are normalized or explicitly documented.
	•	Measurement boundaries are explicit and stable.
	•	Layer 2/3/4 contamination is excluded from the canonical fast-path benchmark.
	•	Diagnostic/instrumentation contamination is excluded from primary measurement runs.
	•	Results are reportable with sufficient metadata for later comparison.
	•	No ambiguity remains about what the canonical fast-path benchmark is measuring.

Phase 2 is not complete if any of the above remains unresolved.

⸻

11. Immediate handoff to Phase 3

Phase 3 may begin only when the following are true:
	•	The new Layer 1 has been benchmark-aligned and validated under the canonical Class A benchmark.
	•	The measured Layer 1 boundary is unambiguous.
	•	Benchmark drift risks are controlled.
	•	Fairness rules versus mlx-lm are frozen for the first implementation cycle.
	•	The benchmark harness can continue to validate the fast path after Layer 2 reattachment.
	•	No unresolved ambiguity remains about what Layer 2 is allowed to add without contaminating the canonical fast-path benchmark.
	•	No unresolved ambiguity remains about how Layer 2 benchmark classes must remain separate from Class A.

Nothing necessary for General Path implementation may remain ambiguous at handoff.