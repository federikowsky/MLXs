# Freeze

Reopen only with new evidence.

## Accepted baselines

- Local accepted baseline: current dirty `refactor/core-exec` worktree.
- Remote accepted baseline:
  - host `llm@169.254.225.109`
  - tree `/tmp/mlxs_layer1_candidate_20260417_1`
  - interpreter `/Users/Shared/mlx-cluster-venv/bin/python`

## Accepted promoted seams

- Sampled-pure short Layer 2 win.
- Compile decode eligibility gate `<=512`.
- Chat optional `prompt_toolkit` boundary fix.
- Batch-host restoration compatibility adapter.
- Qwen flat-config metadata compatibility fix.
- Qwen long-prompt decode benchmark-surface fix.
- Llama-3.2-3B long-decode benchmark-surface fix.
- Shared Layer 1 + Layer 3 aligned plain-KV batch progression kernel.

## Materially closed families

- General-path short enriched `logprobs/top_logprobs` shaping overhead:
  - closed without patch
  - strongest raw-logit shaping probe was weak/noisy
- AC2 scheduler micro-families:
  - merge/scatter elimination
  - grouped greedy fast path
  - resident active batch progression
  - async next-step progression
  - batch-aware cache contract
  - lightweight per-step event materialization
- AC2 batch-host restoration:
  - closed as structural cleanup
  - not a throughput lever

## Process-level freezes

- Mini-family and mini-refactor default behavior for MLXs runtime/performance work:
  - frozen
  - reopen only with explicit file-backed evidence that the front is still only in bottleneck-confirmation mode
- Long same-seam `Rxx-S1/S2/S3/...` chains:
  - frozen by default
  - reopen only with explicit file-backed justification
- Multiple parallel runtime/refactor fronts as the default execution mode:
  - frozen
  - allow only one main front plus optional directly supporting research/instrumentation
- Stale worktree / branch accumulation:
  - frozen by default
  - closed clean paths must be removed unless there is a specific file-backed reason to retain them
  - dirty retained paths must record why deletion is unsafe

## Blocked-for-now

- AC13 prefill / TTFT / RSS superiority on current accepted baseline.
  - best fair eager prefill ratio `1.02808x`
  - target `>=1.05x`
  - bounded local candidates exhausted

## Reopen rules

- Do not reopen any frozen seam because it is nearby or familiar.
- Reopen only with one of:
  - new authoritative remote evidence
  - architecture redesign that changes the relevant ownership/contract
  - cross-model evidence that invalidates the old rejection reason

## Program hold

- No new performance implementation slice until `M01.R1 deep architecture research` is complete and recorded in the file-backed system.

## Research close rule

- `M01.R1` is now complete.
- Resume implementation only against the ranked redesign thesis and the first bounded slice named in `RESEARCH.md`.
- `M01.R1b` is now complete.
- Resume implementation against `M01.R2` unless new evidence invalidates the thesis.

## Rejected redesign shapes

- Class-based resident `DecodeProgression.step()` on the lookahead-sensitive benchmark path.
  - Reason: remote candidate regressed accepted Llama 1B `256` and Qwen `256/2048`.
  - Reopen only with new evidence that removes the hot-loop overhead concern.
- Data-only resident progression-state container on the benchmark helper path.
  - Reason: after fixing the local metrics bug, accepted-sensitive local guardrails still regressed materially.
  - Reopen only with evidence that the container itself is not on the hot loop or no longer affects lookahead-sensitive cases.
- Inline resolved-step helper migration across both benchmark decode branches.
  - Reason: mixed local result; Qwen `2048` remained regressive on focused confirm.
  - Reopen only with evidence that the remaining loss is outside the lookahead branch or that a narrower branch-only migration is safe.
- Broad `general_path` prepared-step branch helper adoption.
  - Reason: mixed local result; Qwen short plain logprobs regressed on focused confirm.
  - Reopen only with evidence that separates enriched top-logprobs leverage from the regressive short plain-logprobs case.
- Narrowed `general_path` enriched top-logprobs-only helper migration.
  - Reason: weak/mixed local result; no strong enough signal to justify remote validation.
  - Reopen only with new evidence that a specific enriched top-logprobs branch has materially higher leverage than the current results show.
- No-patch resident active completion-batch object slice.
  - Reason: regressive on canonical AC2 local truth-first probe and parity-broken.
  - Reopen only with new evidence that fixes parity and changes the throughput direction materially.
- Resident prefill-to-completion handoff for aligned cohorts.
  - Reason: parity held, but canonical AC2 local truth-first compare was still weak/mixed (`256` regressive, `2048` only slightly positive).
  - Reopen only with new evidence that changes the batch contract more broadly than a one-step scheduler handoff.
- Narrow batch-side entry slices as the primary strategy.
  - Reason: both `resident active completion-batch object` and `resident prefill-to-completion handoff` are now rejected.
  - Reopen only as subcomponents of a broader integrated batch-contract redesign, not as standalone performance bets.
- Private Layer 3-only resident fast-batch contract for the canonical AC2 surface.
  - Reason: local result was strong, but remote-authoritative validation split sharply by prompt length (`256` strong positive, `2048` strong regression).
  - Reopen only with new evidence that changes shared Layer 1 / Layer 3 progression ownership or explains the long-prompt regression materially.
- `R17` run_greedy-only short-prompt materialization contract alignment.
  - Reason: local `run_greedy`-only signal was positive, but the current authoritative AC1 surface still runs through the benchmark helper, so the slice did not produce a clean promotable accepted-surface win.
  - Reopen only as one component of a shared helper/core short-prompt slice, not as a standalone `run_greedy`-only bet.
- `R18` shared generator-style short-prompt helper/core convergence slice.
  - Reason: local truth-first comparison was exact in token parity but sharply mixed in performance, with the weakest small-model short-prompt case regressing too hard to justify remote validation.
  - Reopen only with a lower-overhead inline/shared-step shape, not with a shared per-token generator loop.
- `R19` lower-overhead shared short-prompt helper/core convergence slice.
  - Reason: local evidence was strong, but the decisive authoritative remote weakest-case surface still failed to improve.
  - Reopen only with new evidence that the next AC1 slice can move the accepted helper surface more directly than the current convergence family.
- `R22` resident token-lookahead helper-contract redesign.
  - Reason: removing prepared-step/result-object churn from the helper lookahead loop stayed too small locally and slightly regressive on the authoritative remote control/candidate compare.
  - Reopen only with new evidence that the remaining weakest-case tax still lives above the model/cache forward-step substrate, or as one bounded component inside a deeper forward-step/cache redesign.
- Broad steady-state `Llama` model-wrapper redesign as the first `R23` move.
  - Reason: early no-patch `R23` reads kept steady-state cached forward near parity; the first stronger signal is at the prompt-tail to first-decode boundary instead.
  - Reopen only if interleaved first-decode boundary probing collapses or new evidence re-points the decisive delta back to broad steady-state model forward.
- Prompt-tail / first-decode boundary as the next standalone single-request redesign family.
  - Reason: `R23` local + remote probes showed the current MLXs total boundary was already better than comparable `mlx_lm`, while the first no-patch integrated boundary variant was materially worse.
  - Reopen only with new authoritative evidence that a different bounded single-request boundary variant improves the full decisive surface rather than just moving isolated component timing around.
- Old AC2 post-`R10` collapse ranking as the basis for current batch prioritization.
  - Reason: repaired direct comparator and current accepted code supersede it.
  - Reopen only as historical context, not as the governing current ranking.
- Batch merge/scatter/event-shaping micro-fixes as the primary `R24` thesis.
  - Reason: current rerank classifies them as incidental relative to resident prompt/generation ownership and cache-substrate breadth.
  - Reopen only if new fast-path decomposition evidence shows they dominate the remaining prompt-`256` gap on the current accepted surface.
- Resident prompt-batch handoff as the first bounded `R24` slice.
  - Reason: `R24-S1` showed the activation boundary is already competitive; the stable slower component is the active-batch first generation step.
  - Reopen only if new evidence from `R24-S2` shows the first-step penalty is downstream of a hidden handoff cost.
- Materialization-only active-batch slice as a standalone `R24` fix.
  - Reason: `R24-S2` improved the isolated first-step micro-surface but regressed the real prompt-`256` scheduler target remotely.
  - Reopen only if a future overlap probe proves how to preserve next-step readiness while reducing materialization cost.
- Narrow overlap/readiness seam as a standalone `R24` fix.
  - Reason: `R24-S3` classified the issue as broader active-batch state lifetime, not one isolated overlap seam.
  - Reopen only if a future state-lifetime redesign isolates a new bounded overlap lever with end-to-end support.
- `R24-S4a resident dual-step _SharedFastBatch owner`.
  - Reason: first concrete candidate inside the broader active-batch state-lifetime family was attempted, failed local truth-first validation, and was reverted.
  - Reopen only if a future owner-boundary redesign produces new evidence that the same shape is no longer the same problem.
- `R25` canonical extendable active-owner family on the current static prompt-`256` canonical surface.
  - Reason: both the token-resident and dual-ready owner variants over per-row-offset batch cache regressed MLXs absolute scheduler throughput materially.
  - Reopen only on a dynamic late-admission surface or with new evidence that preserves the aligned hot path while adding extendability.
- Per-row-offset batch cache as a required component of the current aligned canonical AC2 fast path.
  - Reason: `R25` local truth-first results show that paying the extendable-cache tax directly on the aligned hot path is the wrong first move.
  - Reopen only if a later design keeps the aligned hot path non-regressive on its own accepted surface.
- `R26-S1` full dynamic late-admission owner as the next first move.
  - Reason: it regressed local staggered throughput materially and broke output parity.
  - Reopen only with new evidence that the dynamic continuation owner can stay exact and avoid the current wall-time penalty.
- `R26-S2` synchronous full prompt prefill inside the main scheduler step.
  - Reason: it fixed late-admission semantics but still missed the wall-clock gate because whole-prompt work stayed on the hot scheduler step.
  - Reopen only if a bounded prompt owner can meter that work without regressing the generation step.
- `R26-S4` scheduler-layer budgeted prompt-owner redesign.
  - Reason: it fixed the late request step positions to near-upstream parity but still missed the throughput gate, which means the remaining tax is below the current Layer 3 prompt-owner loop.
  - Reopen only if a future substrate redesign changes the lower-level prompt-progress cost model materially.
- `R26-S5` lower-layer prompt-progress substrate redesign.
  - Reason: it preserved the corrected step positions but still did not produce a strong enough throughput win.
  - Reopen only with materially new evidence that a different batch family, not this prompt-progress family, is now decisive.
- `R26` batch prompt-progress family for this phase.
  - Reason: the family fixed semantics but did not deliver a promotable throughput lever after the final acceptable descent.
  - Reopen only with materially new evidence.
- Parallel fast generation owners plus prompt-prefill handoff on the staggered surface.
  - Reason: the first integrated `R30` runtime candidate repaired late-request step positions but still lost badly on the real surface because it duplicated generation stepping instead of extending one active owner.
  - Reopen only with new evidence that parallel owners do not materially duplicate active generation work.
- One active extendable dynamic generation owner on the staggered surface.
  - Reason: the last acceptable `R30` candidate preserved exact parity and repaired the late-request step positions, but still failed the real staggered throughput gate strongly enough to close the family.
  - Reopen only as part of a broader substrate redesign that changes mixed-offset cache/mask/rope ownership materially.
- `R30` generation-owner / eval-discipline family for this phase.
  - Reason: the family exhausted its last acceptable attempt without producing a promotable staggered-surface result.
  - Reopen only with materially new authoritative evidence.
- `R28` canonical single-request progression contract refactor family.
  - Reason: removing benchmark-local progression duplication did not improve the accepted AC1-sensitive surfaces enough and regressed Qwen short prompt materially.
  - Reopen only with materially new evidence that another contract-level ownership rewrite is still the right AC1 lever.
- `R29` AC1 lower-level substrate and capability front for this phase.
  - Reason: the no-patch accepted-sensitive local probe kept exact parity but did not isolate one dominant lower-level internal or substrate-level bottleneck across the surviving AC1 surfaces.
  - Reopen only with materially new authoritative evidence that makes one AC1 component clearly dominant again.
- Current `Llama-3.2-1B` and `Qwen2.5-1.5B` benchmarked surfaces as model-mismatch explanations.
  - Reason: focused current-target audits are clean enough that model mismatch is not the leading explanation for the present AC1 gap.
  - Reopen only with new evidence such as non-zero controlled logit/layer divergence, metadata/projection mismatch, or cache-offset inconsistency.
