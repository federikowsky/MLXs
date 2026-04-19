# M01 Risks

## Open

- Runtime redesign may require contract changes across both benchmark helper and `runtime_core`.
- A redesign that helps AC2 may hurt accepted single-request guardrails if state ownership widens carelessly.
- Current accepted wins on Qwen and Llama 3B long decode are benchmark-surface scoped; migrating them into runtime may expose new regressions.
- Remote benchmark host is a single authoritative surface; incorrect remote state or concurrent runs can invalidate evidence.
- `batch.scheduler` currently owns its own decode/materialization/event-shaping loop; redesign must avoid forcing Layer 1 to emit product-rich events just to make batching reusable.
- `general_path` currently duplicates several prepared-step loops; redesign may tempt an overly broad unification that mixes Layer 1 progression with Layer 2 semantics.
- Hot-loop Python dispatch sensitivity is now proven relevant on the lookahead path.
- Even a data-only resident progression container can regress accepted-sensitive cases if it still sits on the per-token hot loop.
- Even inline helper reuse can be mixed if it changes both decode branches together and the remaining divergence lives specifically in the long lookahead branch.
- The promoted `M01.5` win is structurally real but small; future slices must move accepted gains closer to core consumers, not just reshuffle benchmark helper code.
- Research risk: without a full MLXs vs `mlx_lm` vs MLX map, further implementation is likely to optimize symptoms rather than the dominant structural tax.
- Addendum risk check: wider MLX capability inventory may tempt opportunistic primitive chasing; keep the ranked thesis unless a capability family clearly displaces it.
- `run_greedy` now has a bounded promoted lookahead path; future work must avoid leaking benchmark-only model gates into the real core without fresh evidence.
- `batch.scheduler` remains a higher-leverage but higher-risk consumer because it still combines progression, batching, cache rebuild, and event shaping.
- `general_path` broad prepared-step migration is now proven too wide; the remaining safe work must narrow to a more specific subpath.
- The narrowed `general_path` top-logprobs-only slice also failed to produce a strong enough signal; further `general_path` narrowing now looks low-leverage for this phase.
- Batch remains the largest open structural surface, but it combines progression, cache ownership, batch filtering, and product-near event shaping in one subsystem; redesign slices must avoid turning that into a broad uncontrolled rewrite.
- Batch candidate ranking is complete; the next risk is keeping the first batch slice narrow enough to avoid swallowing prompt-processing, queue admission, and product-surface concerns together.
- The first-ranked resident completion-batch slice is now proven regressive and parity-broken; the fallback path must not assume completion-state residency alone is enough.
- The resident completion-batch slice must avoid silently reintroducing the previously rejected scheduler micro-fix families under a new wrapper shape.
- The aligned resident prefill-to-completion handoff slice preserved parity but still failed to produce a strong AC2 local win; a one-step scheduler handoff alone now looks too weak as an entry slice.
- After `R6` and `R7`, the next batch move likely needs a broader ownership shift across prompt-processing and active completion state together; the risk is over-broadening into product-surface or queue-policy changes.
- `R9` introduces a private batch-aware cache/runtime substrate that MLXs does not currently have; the main risk is breaking cache ownership or final-cache extraction while chasing throughput.
- The canonical fast-path restriction is deliberate; the risk is accidentally widening unsupported cache families or imported-cache requests into the new path before correctness and performance are proven.
- If the new private resident fast batch still cannot materially improve AC2, the next remaining leverage likely sits in shared Layer 1 + Layer 3 progression ownership, not another Layer 3-only variant.
- `R9` now confirms a stronger risk: a Layer 3-only batch redesign can look excellent locally and still fail remotely on long prompts. Future batch work must not treat short-prompt wins as sufficient evidence.
- The remote accepted tree lags the accepted local batch surface; authoritative batch validation may need fresh remote mirrors until the official remote baseline is reconciled.
- `R10` must not recreate the `R9` mistake of leaving batched prompt/decode materialization in Layer 3; if Layer 1 still does not own those boundaries, the redesign is not actually testing the shared-ownership thesis.
- `R10` will likely require new private Layer 1 batched progression helpers and row-level cache operations; the risk is widening Layer 1 too far into row lifecycle or product semantics.
- `R10` is now promoted; the new risk is over-reading the scheduler-level AC2 win before re-establishing absolute acceptance against `mlx_lm` on the promoted baseline.
- The scheduler-level AC2 probe is now repo-native tooling; future misuse risk is treating it as the only benchmark surface rather than one input into broader acceptance closure.

## Watch

- Do not let M01 collapse layer boundaries to chase a narrow speedup.
- Do not generalize model-scoped wins without cross-model guardrail evidence.
- Do not reopen blocked AC13 work before the redesign actually changes prefill/runtime ownership.

## Current blocker summary

- No blocker to research.
- `M01.R1` and `M01.R1b` are complete.
- `M01.R2` is complete.
- `M01.R3` consumer ranking is complete and the broad `general_path` slice is rejected.
- `M01.R4` narrowed `general_path` slice is complete and rejected.
- `M01.R5` is complete.
- `M01.R6` is complete and rejected.
- `M01.R7` is complete and rejected.
- `M01.R8` ranking is complete.
- `M01.R9` is complete and rejected.
- `M01.R10` is complete and promoted.
- Next blocker is re-ranking acceptance closure from the new promoted baseline before choosing the next redesign front.
