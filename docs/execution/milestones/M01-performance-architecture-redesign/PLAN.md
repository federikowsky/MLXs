# M01 Plan — Performance Architecture Redesign

## Goal

Replace exhausted micro-fix loops with structural runtime redesign that can materially improve AC1 and AC2 versus `mlx_lm`.

## Current mode

- `M01.PC1 repository process correction — autonomous integrated-refactor policy hardening` is complete.
- `R26` prompt-progress batch family is now closed for this phase.
- `R29` AC1 substrate rerank is complete and closes the current AC1 front.
- `R30` is now closed for this phase after its last acceptable attempt failed locally.
- Current active path:
  - `M01.R31 mixed-offset batch cache substrate redesign on the staggered surface`

## Entry evidence

- AC1 remains open on broadened accepted set even after Qwen and Llama 3B benchmark-surface wins.
- The repaired AC2 direct surface is no longer catastrophic:
  - prompt `256 requests/s ~0.99661x`
  - prompt `2048 requests/s ~1.09404x`
- AC13 is blocked-for-now on the current accepted baseline.
- `R16` local truth-first decomposition shows the benchmark/core split is still material on short-prompt cases.
- `R17` proved that a `run_greedy`-only slice is too narrow to close the accepted AC1 surface.
- `R18` proved that a shared per-token generator loop is too expensive on the weakest small-model short-prompt case.
- `R19` proved that even a lower-overhead shared helper/core convergence slice did not move the decisive weakest-case remote surface.
- `R22` proved that even a resident token-lookahead helper contract does not materially improve the decisive weakest-case remote surface.
- `R23` completed local + remote substrate probes and did not expose a bounded promotable single-request redesign slice.
- `R24` rebuilt the current batch divergence map from the repaired direct comparator, not from stale post-`R10` beliefs.
- `R24-S1` showed current fast-path activation is already competitive; the stable slower component is the first active-batch generation step.
- `R24-S2` proved that one-step materialization is not an isolated lever; removing grouped current+next eval regressed the real prompt-`256` scheduler target.
- `R24-S3` proved that overlap/readiness is real but belongs to broader active-batch state lifetime rather than a narrow overlap seam.
- `R24-S4a` is already attempted and rejected; the next move must use a new path identifier and a different concrete thesis.
- `R25` token-resident extendable owner local scheduler control/candidate `requests_per_s ~0.79339x`.
- `R25` dual-ready extendable owner local scheduler control/candidate `requests_per_s ~0.73350x`.
- new staggered local direct compare shows the larger dynamic extension gap:
  - prompt `256 mlxs/mlx_lm requests_per_s ~0.69428x`
- authoritative remote staggered baseline now confirms the current accepted gap:
  - prompt `256 mlxs/mlx_lm requests_per_s ~0.85088x`
- authoritative staggered decomposition now proves late-request starvation:
  - MLXs late first-token step `129` vs `mlx_lm 4`
  - MLXs late completion step `255` vs `mlx_lm 131`
- `R26-S1` full dynamic late-admission owner is rejected.
- `R26-S2` synchronous late prefill plus legacy continuation handoff is rejected as neutral-to-regressive even though it fixes the step-level admission semantics.
- `R26-S4` proves the budgeted prompt-owner semantics are sufficient:
  - late first-token step `4`
  - late completion step `130`
- `R26-S4` also proves the remaining loss is below the current prompt owner:
  - requests/s `candidate/control ~0.91001x`
  - p95 completion `~1.14951x`
- `R30` local no-patch lower-boundary probe now narrows the remaining tax further:
  - exact output parity `True` on two local seeds
  - MLXs late first-token/completion steps stay `129/255`
  - `mlx_lm` late first-token/completion steps stay `4/131`
  - MLXs lower-boundary time is dominated by `mx.eval` at the fast-batch and grouped-decode boundary
  - `mlx_lm` lower-boundary time is much more heavily shifted into `mx.async_eval`
  - attention, mask, and cache-update timings stay tiny relative to the evaluation-boundary split
- Focused current-target model audits now say the current `Llama` and `Qwen` implementations look correct enough for this target.
- Many adjacent seam families are already tested and rejected.
- first integrated `R30` runtime candidate result:
  - repaired late-request steps to `4/131`
  - local staggered requests/s still only `~0.70730x`
  - lower-boundary probe shows the candidate duplicated generation stepping instead of extending one active owner
- last acceptable `R30` candidate result:
  - preserved exact parity and the repaired `4/131` steps
  - local staggered requests/s still only `~0.72761x`
  - `R30` is therefore closed as a family

## Redesign hypothesis

The `AC1` front is now closed for this phase after `R29`.

The process itself is also now corrected:

- no more mini-family / mini-refactor default behavior
- small-step work is only for bottleneck confirmation
- once a front is sufficiently understood and has burned about `2-3` bounded families without a strong win, the next move must be integrated refactor or closure/rerank

The strongest remaining path is now broader than owner lifetime alone:

- mixed-offset batch cache substrate redesign on the staggered-only path
- specifically:
  - heterogeneous-offset cache representation
  - mask construction
  - rope / model-boundary behavior
  - preserve the aligned static hot path
  - no standalone built-in-MLX or custom-kernel pivot unless later evidence isolates a true op-level bottleneck

## Proposed fronts

1. `F1 mixed-offset batch cache substrate redesign`
   - broader redesign beyond the closed `R30` generation-owner family
2. `F2 close AC2 and rerank globally`
   - only if the broader substrate thesis still cannot expose a bounded viable path

## Chosen entry front

- `F1 mixed-offset batch cache substrate redesign`
- Reason:
  - `R30` exhausted its last acceptable owner-lifetime attempt
  - the failed final candidate implies the blocking issue is broader cache/mask/rope substrate cost on the dynamic path
  - built-in MLX and custom-kernel pivots remain unjustified
- Immediate consequence:
  - open a fresh dedicated worktree for `R31`
  - define the bounded `R31` thesis before any runtime patch
  - do not pivot to standalone built-in-MLX or custom-kernel work on current evidence

## Target boundary

The next redesign target is below the closed prompt-owner / prompt-progress family on the staggered batch surface:

- batched forward-step structure
- batched cache update/fetch behavior
- batched attention/mask construction
- eval / async-eval ordering under active late admission
- stream ownership at the lower batch execution boundary

It must not reopen:

- `R26` scheduler-layer prompt-owner / prompt-progress variants
- `AC1` helper/core or contract-level families
- AC13 prefill work

## Initial sequencing

1. apply repo-level process correction
2. normalize file-backed state so the corrected policy is binding
3. normalize worktree / branch hygiene and remove safe stale paths
4. run the local `R30` no-patch lower-boundary probe
5. continue immediately into the chosen integrated lower-level batch path

## Research sequencing

1. framing
2. source archaeology
3. architecture maps
4. divergence map
5. bottleneck model
6. ranked redesign thesis
7. first bounded implementation slice definition
8. reject or promote from the staggered truth-first package

## Non-goals

- no new product-surface redesign
- no speculative benchmark breadth expansion before redesign evidence demands it
- no reopening frozen helper-only micro-fix seams without new evidence
- no AC13 reopening work on this path
- no more attempts to make the aligned canonical fast path itself carry the extendable-cache tax as the first move
- no more synchronous whole-prompt prefill inside one live generation scheduler step
- no more scheduler-layer budget tuning under the same `R26-S4` shape
- no more prompt-progress batch variants under `R26`
- no more contract-level AC1 ownership rewrites under the rejected `R28` family by default
- no more batch prompt-owner / prompt-progress variants under the closed `R26` family by default
