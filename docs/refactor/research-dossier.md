MLXs Research Dossier

Purpose: lock the technical constraints of MLX and the current mlx-lm baseline before further architectural refinement.
Scope: only verified facts, reasonable inferences, and open questions relevant to the new MLXs core.
Source priority: official MLX documentation first; current mlx-lm public code/docs second.

1. MLX execution model

1.1 Claim

MLX uses lazy execution: operations record a compute graph, and computation happens only when an evaluation boundary is reached.
Status: Verified
Evidence: The MLX lazy-evaluation docs state that when you perform operations in MLX, “no computation actually happens,” a compute graph is recorded, and actual computation happens when eval() is performed. The quick-start docs also say outputs are not computed until needed.  ￼
Architectural consequence: The MLXs core must treat graph construction and graph evaluation as distinct phases. API and internal contracts must make evaluation boundaries explicit instead of hiding them behind wrappers or rich objects.
Need for benchmark or experiment: No, for the existence of lazy execution. Yes, for quantifying graph-build cost under MLXs-specific decode workloads.

1.2 Claim

Some operations trigger implicit evaluation: array.item(), printing an array, and converting an MLX array to numpy.ndarray.
Status: Verified
Evidence: The MLX quick-start docs explicitly say that array.item(), printing an array, and converting to NumPy all automatically evaluate the array.  ￼
Architectural consequence: The performance core must forbid accidental host-side materialization in hot paths. Debug prints, scalar reads, and NumPy conversions are not neutral operations; they are evaluation boundaries.
Need for benchmark or experiment: No, for semantics. Yes, if MLXs wants to quantify the exact cost of specific materialization patterns in decode loops.

1.3 Claim

Unused outputs may avoid device computation, but their graphs are still built and graph construction has a cost.
Status: Verified
Evidence: The lazy-evaluation docs say that if one output is unused, the expensive branch is not actually computed, but the graph for that branch is still built and “has some cost associated to it.”  ￼
Architectural consequence: “Returning extra things for flexibility” is not free even when callers ignore them. The fast path should not construct optional outputs unless they are required by that path.
Need for benchmark or experiment: Yes. For MLXs specifically, this should be measured for returning logits, logprobs, metadata, and auxiliary tensors in the decode step.

1.4 Claim

Each graph evaluation has fixed overhead, while extremely large graphs also incur overhead that grows with graph size.
Status: Verified
Evidence: The MLX lazy-evaluation docs state there is fixed overhead with each graph evaluation and some overhead that grows with compute-graph size; they add that a broad range of graph sizes works well, from tens to many thousands of operations per evaluation.  ￼
Architectural consequence: MLXs should not adopt either extreme as dogma. “Evaluate every tiny step” and “let graphs grow arbitrarily” are both unsupported. The core needs an explicit evaluation policy tuned to the decode workload.
Need for benchmark or experiment: Yes. This directly motivates benchmark work for decode-step boundary placement, prefill chunk size, and any periodic cache-clearing policy.

1.5 Claim

mx.eval() evaluates arrays or trees of arrays; non-array leaves are ignored.
Status: Verified
Evidence: The MLX eval docs say eval(*args) evaluates an array or tree of arrays, where trees can be Python lists, tuples, or dicts, and non-array leaves are ignored.  ￼
Architectural consequence: Batch evaluation of multiple arrays is supported directly. MLXs should prefer bulk evaluation at explicit boundaries over fragmented, repeated scalar materializations.
Need for benchmark or experiment: Yes. This supports testing “one grouped eval” versus multiple item() or fragmented eval calls in the same logical step.

1.6 Claim

MLX can export the unevaluated graph rooted at given outputs for visualization.
Status: Verified
Evidence: mlx.core.export_to_dot exports a graph to DOT and recursively includes all unevaluated inputs of the provided outputs.  ￼
Architectural consequence: Graph-shape and graph-size questions are investigable from Python without relying only on wall-clock behavior. MLXs can use graph export as part of research and regression tooling.
Need for benchmark or experiment: Yes. This is a candidate tool for comparing fast-path graph shape between MLXs and mlx-lm.

⸻

2. MLX stream semantics

2.1 Claim

All MLX operations, including random number generation, accept an optional stream argument.
Status: Verified
Evidence: The official “Using Streams” docs state that all operations, including random number generation, take an optional stream keyword argument.  ￼
Architectural consequence: Stream discipline is a first-class runtime design concern, not a small optimization detail. Any sampling path is also part of stream design because RNG follows stream assignment.
Need for benchmark or experiment: No, for semantics. Yes, for measuring single-stream vs multi-stream runtime behavior in MLXs.

2.2 Claim

If no stream is specified, operations run on the default stream of the default device.
Status: Verified
Evidence: The same MLX docs state that if stream is unspecified, the operation runs on mx.default_stream(mx.default_device()).  ￼
Architectural consequence: “Default stream” is an actual runtime choice, not an absence of choice. MLXs must decide when to rely on default-stream behavior and when to make stream selection explicit.
Need for benchmark or experiment: Yes. Default-stream versus explicit-generation-stream behavior should be benchmarked, not assumed.

2.3 Claim

Passing a device as the stream argument means the operation runs on the default stream of that device.
Status: Verified
Evidence: The streams docs state that stream can also be a device, in which case the operation runs on the default stream of the provided device.  ￼
Architectural consequence: MLXs should avoid ambiguous stream/device-routing helpers in the core. Stream ownership should stay explicit to keep scheduling behavior inspectable.
Need for benchmark or experiment: No immediate benchmark need unless MLXs wants multi-device support later.

2.4 Claim

MLX provides explicit synchronization primitives for streams.
Status: Verified
Evidence: The devices-and-streams docs expose synchronize([stream]) to synchronize with a given stream.  ￼
Architectural consequence: MLXs can model synchronization explicitly at stream boundaries instead of relying only on implicit host reads. This makes sync strategy a designable, testable part of the runtime.
Need for benchmark or experiment: Yes. MLXs should compare explicit stream synchronization against scalar-read-driven synchronization on the fast path.

2.5 Claim

A single-stream runtime is simpler to reason about, while a multi-stream runtime expands the scheduling state space because compute and RNG can both be stream-scoped.
Status: Inferred
Evidence: Official MLX docs confirm stream-scoped execution and stream-scoped RNG; they do not prescribe single-stream or multi-stream architecture.  ￼
Architectural consequence: MLXs should treat multi-stream execution as an optimization domain requiring explicit proof, not as a default assumption for the new core.
Need for benchmark or experiment: Yes. This requires targeted experiments for prefill/decode overlap, RNG consistency, and hidden synchronization risk.

⸻

3. MLX compile constraints

3.1 Claim

mx.compile() compiles computation graphs and can reduce graph size by merging common work and fusing operations, improving runtime and memory use in many cases.
Status: Verified
Evidence: The compilation docs say compile() compiles computation graphs, resulting in smaller graphs by merging common work and fusing certain operations, which can produce big runtime and memory improvements.  ￼
Architectural consequence: Compile is a real optimization lever for MLXs, but only where the function contract is compatible with MLX’s compile model.
Need for benchmark or experiment: Yes. Any proposed compile seam for MLXs must be benchmarked against the eager baseline.

3.2 Claim

Compiled functions are intended to be pure and should not have side effects.
Status: Verified
Evidence: The MLX docs explicitly state that compiled functions are intended to be pure and should not have side effects; their side-effect example crashes.  ￼
Architectural consequence: A decode core whose correctness depends on implicit side effects is a poor compile target. MLXs must not design compile-first around hidden state mutation.
Need for benchmark or experiment: No, for the constraint itself. Yes, for testing whether a proposed decode-step contract can remain pure enough.

3.3 Claim

The first call to a compiled function can be relatively slow because MLX builds the graph, optimizes it, and generates and compiles code; compiled functions are then cached.
Status: Verified
Evidence: The docs state that the first call builds the compute graph, optimizes it, and generates and compiles code, which can be relatively slow; MLX caches compiled functions afterward.  ￼
Architectural consequence: Compile is suitable only for repeatedly reused functions. MLXs must not rely on compile for one-off or shape-fragmented paths. Warmup and amortization are part of any honest benchmark protocol.
Need for benchmark or experiment: Yes. MLXs needs compile warmup policy benchmarks and must separate cold-start from steady-state measurement.

3.4 Claim

Changing input shape, number of dimensions, input type, or number of inputs can trigger recompilation.
Status: Verified
Evidence: The compile docs list these cases as reasons a function may be recompiled.  ￼
Architectural consequence: A compiled decode core is only viable if its input contract is shape-stable and type-stable, or if shapeless compilation is demonstrably safe and effective for that contract.
Need for benchmark or experiment: Yes. This is a blocking benchmark area for any compiled decode-step design.

3.5 Claim

Compiled functions treat non-parameter-list inputs as constants unless state is passed explicitly or captured via inputs=.
Status: Verified
Evidence: The docs show that state not in the parameter list is treated as constant, and changes are not reflected unless state is passed as an input or captured via inputs=.  ￼
Architectural consequence: Any decode design that closes over mutable cache state or mutable generation state must either expose that state in the compile contract or avoid compile.
Need for benchmark or experiment: Yes. This directly affects whether cache state can be safely part of a compiled step.

3.6 Claim

MLX supports capturing implicit outputs via outputs= and can use inputs= plus outputs= for stateful compiled training steps.
Status: Verified
Evidence: The compile docs show outputs= for implicit outputs and a training-step example using both inputs= and outputs= to capture model, optimizer, and random state.  ￼
Architectural consequence: State threading through compiled functions is supported, but only with explicit discipline. MLXs should not assume that this makes arbitrary decode-state management cheap or simple.
Need for benchmark or experiment: Yes. MLXs needs focused experiments on whether such state threading is practical and performant for autoregressive decode state.

3.7 Claim

If a compiled function relies on random-sampling modules, random state must also be part of captured state.
Status: Verified
Evidence: The compile docs explicitly note that if a module performs random sampling, mx.random.state should be included in the captured state.  ￼
Architectural consequence: Sampling inside a compiled decode path is not a trivial add-on. RNG state becomes part of the correctness contract.
Need for benchmark or experiment: Yes. This is a design-blocking experiment for compiled sampled decode.

3.8 Claim

Shapeless compilation avoids recompilation on shape changes, but static-shape assumptions inside the function can still break correctness.
Status: Verified
Evidence: The docs say shapeless=True avoids recompilation for shape changes, but their example fails when the function embeds static shape assumptions and succeeds only after rewriting to avoid them.  ￼
Architectural consequence: Shapeless compilation is not automatically safe for variable-shape decode logic. MLXs would need a function body explicitly written to avoid static-shape assumptions.
Need for benchmark or experiment: Yes. This remains an open compatibility question for any compiled decode core.

3.9 Claim

compile() is compatible with a decode core only when the decode-step contract is narrow, repeatedly reused, state-disciplined, and shape-stable or safely shapeless.
Status: Inferred
Evidence: This follows from the official compile constraints on purity, state handling, recompilation triggers, caching, and shapeless caveats.  ￼
Architectural consequence: MLXs should not freeze a compile-first decode architecture until a concrete decode-step contract satisfies those constraints empirically.
Need for benchmark or experiment: Yes. This is a design-freeze blocker.

⸻

4. Trusted MLX substrate

4.1 Claim

mlx.core.fast.scaled_dot_product_attention is a native fast attention primitive supporting MHA, GQA, and MQA, with softmax performed in float32 regardless of input precision.
Status: Verified
Evidence: The official API docs describe fast.scaled_dot_product_attention, list supported attention forms, and note that softmax is performed in float32.  ￼
Architectural consequence: This belongs in the trusted substrate for the MLXs core. The core should prefer MLX-native fast attention over rebuilding attention behavior in Python.
Need for benchmark or experiment: Yes. MLXs should still benchmark its usage pattern and mask/layout choices, but the primitive itself is an approved baseline candidate.

4.2 Claim

MLX exposes a “fast” family including RMS norm, layer norm, RoPE, scaled dot-product attention, and Metal kernel support.
Status: Verified
Evidence: The MLX “Fast” docs list rms_norm, layer_norm, rope, scaled_dot_product_attention, and metal_kernel.  ￼
Architectural consequence: MLXs should treat these as preferred implementation substrate when they match the core’s needs. Re-implementing equivalent functionality in Python should require explicit justification.
Need for benchmark or experiment: Yes. Primitive choice is still subject to end-to-end decode benchmarks, but these are the default candidates.

4.3 Claim

MLX supports custom Metal kernels and custom extensions, but these are explicit extension mechanisms rather than default runtime substrate.
Status: Verified
Evidence: MLX documents both fast.metal_kernel and custom extensions for CPU/GPU operations.  ￼
Architectural consequence: MLXs can escalate to custom kernels or extensions if needed, but this should not be assumed in the initial performance-core spec.
Need for benchmark or experiment: Yes. This is an escalation path, not a starting assumption.

4.4 Claim

Patterns that force frequent evaluation boundaries, implicit host materialization, or unnecessary NumPy conversion are performance-risk patterns for the core.
Status: Inferred
Evidence: MLX docs verify lazy execution, implicit evaluation by item()/print/NumPy conversion, and fixed overhead per graph evaluation.  ￼
Architectural consequence: The MLXs performance core should avoid:
	•	debug prints of MLX arrays,
	•	eager conversion to NumPy,
	•	scattered item() reads,
	•	fragmented eval patterns.
Need for benchmark or experiment: Yes. The exact decode impact should be quantified.

4.5 Claim

Graph export is part of the trusted research substrate for MLXs, even if not part of the shipping runtime.
Status: Inferred
Evidence: MLX officially supports exporting unevaluated graphs rooted at outputs.  ￼
Architectural consequence: MLXs should reserve a diagnostics/tooling surface for graph inspection when investigating performance regressions or compile compatibility.
Need for benchmark or experiment: Yes. This should be part of the research and regression toolkit.

⸻

5. Competitive baseline: mlx-lm

5.1 Claim

mlx-lm is an official Apple Silicon MLX package for generation and fine-tuning, with Python API, CLI, and server surfaces.
Status: Verified
Evidence: The repo README says mlx-lm is a Python package for generating text and fine-tuning large language models on Apple silicon with MLX, and documents its CLI and Python API.  ￼
Architectural consequence: MLXs should distinguish clearly between:
	•	baseline features that matter for direct runtime comparison,
	•	broader product surfaces that matter for parity but not for fast-path benchmarking.
Need for benchmark or experiment: No.

5.2 Claim

The current comparable mlx-lm generation path creates a dedicated generation stream, builds prefill and decode work under with mx.stream(generation_stream), uses mx.async_eval(y, logprobs) in decode, materializes the seed with mx.eval(y), yields y.item() with logprobs, and calls mx.clear_cache() every 256 decode iterations.
Status: Verified
Evidence: The current mlx_lm/generate.py shows:
	•	generation_stream = mx.new_stream(mx.default_device()),
	•	_step() running model work under with mx.stream(generation_stream),
	•	mx.async_eval(y, logprobs) after the first step and for subsequent steps,
	•	mx.eval(y) at n == 0,
	•	yield y.item(), logprobs,
	•	if n % 256 == 0: mx.clear_cache().  ￼
Architectural consequence: Any MLXs benchmark claiming to compare against mlx-lm fast-path generation must account for these baseline runtime choices instead of treating mlx-lm as a black box.
Need for benchmark or experiment: Yes. MLXs must benchmark against matching workload conditions and document any deliberate divergences.

5.3 Claim

The mlx-lm generation path supports prompt-cache creation, prompt-cache in-place updates, quantized KV start, logits processors, and speculative decoding.
Status: Verified
Evidence: The current generate.py documents prompt-cache creation and in-place update, KV quantization options, logits_processors, and speculative_generate_step with draft model and drafted token count.  ￼
Architectural consequence: These features are part of mlx-lm’s broader runtime surface, but they should not automatically be included in the MLXs fast-path benchmark.
Need for benchmark or experiment: Yes. MLXs should isolate which of these belong in separate benchmark tracks.

5.4 Claim

The mlx-lm server exposes OpenAI-like chat/completions behavior and supports streaming, sampling controls, repetition/presence/frequency penalties, logit bias, logprobs, and speculative decoding through draft_model and num_draft_tokens.
Status: Verified
Evidence: SERVER.md documents the HTTP model server, request fields including stream, temperature, top_p, top_k, min_p, repetition/presence/frequency penalties, logit_bias, logprobs, draft_model, and num_draft_tokens.  ￼
Architectural consequence: MLXs should keep server and runtime feature parity on the roadmap, but server-surface breadth must be benchmarked separately from the fast-path decode kernel.
Need for benchmark or experiment: No, for feature existence. Yes, for deciding which server/runtime features form separate benchmark suites.

5.5 Claim

A correct competitive workload for the fast path is one where both MLXs and mlx-lm perform the same model-level task with aligned generation settings and comparable runtime boundaries.
Status: Inferred
Evidence: This follows from the verified mlx-lm generation path and server/runtime feature set. The official docs do not define a benchmark protocol, but the current path includes explicit stream use, async_eval, scalar materialization, and periodic cache clearing.  ￼
Architectural consequence: The fast-path benchmark should be restricted to workload slices that are behaviorally comparable:
	•	same model,
	•	same prompt tokenization,
	•	same sampling regime,
	•	same output target,
	•	same or explicitly normalized cache-clearing policy,
	•	no unrelated server-side overhead.
Need for benchmark or experiment: Yes. This should become part of the MLXs benchmark protocol.

5.6 Claim

Features that materially widen work per token or per request should stay out of the fast-path benchmark unless the benchmark is explicitly defined to include them.
Status: Inferred
Evidence: mlx-lm supports penalties, logprobs, speculative decoding, prompt-cache manipulation, and server features beyond minimal decode. These are verified feature surfaces, but not all belong to the same benchmark objective.  ￼
Architectural consequence: MLXs should define separate benchmark classes for:
	•	minimal decode fast path,
	•	enriched single-request generation,
	•	speculative,
	•	prompt-cache reuse,
	•	server/streaming.
Need for benchmark or experiment: Yes. This is required to prevent misleading mixed-workload comparisons.

⸻

6. Architectural consequences derived from the evidence

6.1 Claim

Evaluation boundaries are architectural boundaries in MLX, not implementation details.
Status: Verified
Evidence: MLX computes lazily and evaluates only when needed; eval(), item(), printing, and NumPy conversion all trigger evaluation.  ￼
Architectural consequence: The MLXs performance core must make evaluation points explicit in both code structure and contracts.
Need for benchmark or experiment: Yes. Decode-step boundary placement remains an empirical tuning area.

6.2 Claim

Returning optional data from the core is not free even if callers ignore it.
Status: Verified
Evidence: MLX docs say unused outputs may avoid actual computation, but their graphs are still built and this has cost.  ￼
Architectural consequence: The fast path should not share the same per-step return contract as feature-rich paths unless the extra outputs are proven negligible.
Need for benchmark or experiment: Yes. This is a benchmarkable design question.

6.3 Claim

Stream policy affects both compute scheduling and RNG behavior.
Status: Verified
Evidence: MLX states that all operations, including random number generation, accept a stream argument.  ￼
Architectural consequence: Sampling design and stream design cannot be treated independently in MLXs.
Need for benchmark or experiment: Yes. Especially for any future multi-stream path.

6.4 Claim

A compiled decode core is only a legitimate design target if its state, RNG, and shape behavior fit MLX’s compile constraints.
Status: Verified
Evidence: This follows from the verified compile rules on purity, state capture, cached compilation, recompilation triggers, and random state capture.  ￼
Architectural consequence: MLXs cannot freeze a compile-first architecture on analogy alone. The compile seam must be validated against MLX’s documented rules first.
Need for benchmark or experiment: Yes. This is a design-freeze blocker.

6.5 Claim

The trusted substrate for the MLXs core should default to MLX-native fast primitives and explicit graph/runtime tooling, not to Python-layer reinvention.
Status: Verified
Evidence: MLX officially documents fast attention, fast norm/rope primitives, graph export, and custom-kernel escape hatches.  ￼
Architectural consequence: MLXs should treat native MLX fast primitives as the baseline implementation substrate for the core, and reserve higher-level wrappers or custom kernels for justified cases.
Need for benchmark or experiment: Yes. Primitive composition and layout choices still need empirical validation.

6.6 Claim

The current mlx-lm baseline is broad enough that MLXs must separate “competitive fast path” from “feature-complete path” to benchmark fairly.
Status: Inferred
Evidence: mlx-lm’s runtime and server surfaces include prompt cache, KV quantization, logits processors, logprobs, speculative decoding, streaming, and penalty options.  ￼
Architectural consequence: The benchmark protocol and later spec work must define explicit workload classes instead of a single blended “generate” benchmark.
Need for benchmark or experiment: Yes. This is required before spec freeze.

⸻

7. Confirmed conclusions from current spec

The following conclusions are already supported strongly enough to keep in the working direction:
	1.	MLXs must treat lazy execution and evaluation boundaries as first-class design constraints, not hidden runtime details.  ￼
	2.	The performance core should avoid accidental host materialization and fragmented evaluation.  ￼
	3.	Stream policy is a real architectural concern because compute and RNG are both stream-scoped in MLX.  ￼
	4.	mx.compile() is real and valuable, but only under strict purity/state/shape discipline.  ￼
	5.	MLX-native fast primitives belong in the trusted substrate of the future MLXs core.  ￼
	6.	mlx-lm already has a broad generation/runtime/server surface, so fast-path benchmarking must be narrower and more explicit than full product parity.  ￼

⸻

8. Weakly supported conclusions that need caution

These are plausible, but not yet strong enough to freeze as hard design rules:
	1.	A single-stream MLXs core is likely the safest starting point, but that is an inference from MLX stream semantics, not a documented prescription.  ￼
	2.	A compiled decode seam may be viable only for a very narrow contract, but this is still contingent on actual MLXs decode-step experiments.  ￼
	3.	Returning extra optional tensors from the fast path is likely costly enough to justify separate return contracts, but the magnitude still needs direct measurement in MLXs.  ￼
	4.	Multi-stream execution may be worthwhile only after the baseline core is stable, but that is a strategy inference, not a fact directly stated by MLX docs.  ￼

⸻

9. Open technical questions that block design freeze
	1.	Decode-step graph size and shape: how large is the unevaluated decode graph in candidate MLXs designs, and how does it compare to the comparable mlx-lm path? Graph export exists, but this has not been measured yet.  ￼
	2.	Best evaluation boundary policy for decode: what grouped eval() / item() strategy minimizes total overhead while keeping the graph compact enough? MLX docs define the tradeoff but not the optimum for this workload.  ￼
	3.	Compile compatibility of a real decode core: can a narrow decode-step contract satisfy MLX compile purity, state, shape, and RNG constraints without architectural distortion?  ￼
	4.	Shapeless compile suitability: if decode-related shapes vary, can shapeless compilation be used safely without static-shape traps?  ￼
	5.	Stream policy for MLXs core: is single-stream measurably best for the initial fast path, or is there a justified multi-stream overlap strategy that improves performance without introducing hidden sync or RNG complexity?  ￼
	6.	Fast-path benchmark protocol: which exact workload class should define competitive comparison with mlx-lm before broader feature suites are considered? The baseline surfaces are known, but the correct benchmark taxonomy still needs to be frozen.  ￼