# Status

Updated: 2026-04-21

## Active milestone

- `M01-performance-architecture-redesign`

## Active workstream

- `M01.R31 mixed-offset batch cache substrate redesign on the staggered surface`

## Strategic objective

- Move beyond exhausted micro-fix families into a redesign capable of producing a stronger practical win over `mlx_lm`.

## Process correction

- The previous runtime/performance process had become too slice-fragmented.
- Repo-level process correction is now applied.
- Future MLXs runtime/performance work must follow:
  - `AGENTS.md`
  - `docs/execution/REDESIGN_MODE.md`
  - `docs/execution/REFACTOR_POLICY.md`
- The corrected rule is:
  - small-step work is only for locating and confirming a bottleneck
  - once a front is sufficiently understood and has burned about `2-3` bounded families without a strong win, the default next move is integrated refactor or closure/rerank
  - no more long same-seam `Rxx-S1/S2/S3` chains by default
  - one main runtime/refactor front at a time
  - risky redesign paths use dedicated worktrees/branches and closed clean paths are cleaned up by default

## Worktree hygiene

- Active baseline worktree:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-core-exec`
  - branch `refactor/core-exec`
  - role: file-backed execution truth plus neutral benchmark/probe tooling
- Active technical worktree retained for the current front:
  - `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r31-mixed-offset-cache-substrate`
  - branch `codex/m01-r31-mixed-offset-cache-substrate`
  - role: dedicated runtime/refactor worktree for the broader post-`R30` staggered redesign thesis
- Cleaned now:
  - pruned stale record `/private/tmp/mlxs_longcase_control`
  - removed worktrees and deleted branches:
    - `codex/m01-r17-ac1-short-prompt-alignment`
    - `codex/m01-r18-ac1-shared-short-prompt`
    - `codex/m01-r19-ac1-inline-short-prompt`
    - `codex/m01-r20-ac1-helper-hotloop`
    - `codex/m01-r21-ac1-schedule-next-contract`
    - `codex/m01-r28-ac1-control`
    - `codex/m01-r28-ac1-integrated-core-refactor`
    - `codex/m01-r30-ac2-substrate-rerank`
- Retained now:
  - dirty historical M01 worktrees `R22`, `R23`, `R24`, `R25`, `R26`, and `R29`
  - reason: each still holds unreviewed local probe/docs state and is not safe for autonomous deletion yet
  - older dirty legacy worktrees outside the current M01 front are retained for the same reason pending a separate hygiene pass

## Priority order

1. core dominance first
2. general-path parity/superiority
3. serving/orchestration strength
4. compiled stabilization
5. product-ready standard

## Accepted baseline summary

- Durable git checkpoint committed: `fcb69de` captures the accepted promoted baseline through `R10` plus the `R11` handoff state.
- Sampled-pure short Layer 2 win promoted.
- Compile decode eligibility promoted with prompt-aware gate `<=512`.
- Chat optional `prompt_toolkit` boundary fix promoted.
- Qwen attention/RoPE metadata compatibility fix promoted.
- Qwen long-prompt decode benchmark-surface fix promoted.
- Llama-3.2-3B long-decode benchmark-surface fix promoted.
- Shared Layer 1 + Layer 3 aligned plain-KV batch progression kernel promoted.
- Repo-native AC2 direct comparator/tooling promoted.
- AC2 scheduler-level baseline exists.
- AC2 batch-host restoration promoted as structural cleanup, not throughput lever.
- Serving/live-stack/operator-visibility families materially closed for this phase.

## Acceptance reality

- `AC1` remains below the program bar on the accepted reference set, but the current AC1 internal/substrate front is now closed for this phase.
- `AC2` remains open and is now the strongest active front:
  - repaired direct surface:
  - `256 requests/s ~0.99661x`
  - `2048 requests/s ~1.09404x`
  - staggered remote surface:
    - `requests/s ~0.85088x`
  - `R30` local lower-boundary probe:
    - exact output parity `True`
    - MLXs late first-token/completion steps `129/255`
    - `mlx_lm` late first-token/completion steps `4/131`
    - attention/mask/cache-update deltas stayed tiny relative to evaluation-boundary timing
  - first integrated `R30` runtime candidate:
    - exact output parity `True`
    - staggered local requests/s `~0.70730x`
    - late request steps repaired to `4/131`
    - rejected because throughput stayed too far below `mlx_lm`
  - last acceptable `R30` candidate:
    - exact output parity `True`
    - staggered local requests/s `~0.72761x`
    - late request steps repaired to `4/131`
    - rejected and `R30` is now closed for this phase
- `AC13` blocked-for-now on current accepted baseline.

## Current accepted reference set

- `mlx-community/Llama-3.2-1B-Instruct-4bit`
  - `256 decode 0.99997x`
  - `2048 decode 1.07037x`
- `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
  - `256 decode 1.02031x`
  - `2048 decode 1.02585x`
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
  - `256 decode 1.01699x`
  - `2048 decode 1.01974x`

None clears `AC1 >= 1.10x decode`.

## Current blockers

- Local seam families are mostly exhausted.
- Current accepted wins on Qwen and Llama 3B long decode are benchmark-surface scoped, not architectural.
- The stale post-`R10` AC2 collapse ledger is superseded by the repo-native direct comparator:
  - `/tmp/mlxs_ac2_direct_compare_candidate_remote_20260419_2.json`
  - exact output parity on both prompt sizes
- `AC1` remains genuinely open on the accepted set:
  - weakest accepted case is still `Llama-3.2-1B 256 decode 0.99997x`
  - no accepted reference clears `AC1 >= 1.10x decode`
- `R16` local helper/core split shows the benchmark/core gap is still material on short-prompt cases:
  - `Llama-3.2-1B 256 core/helper decode ratio ~0.8212`
  - `Qwen2.5-1.5B 256 core/helper decode ratio ~0.7911`
  - `Llama-3.2-1B 2048 core/helper decode ratio ~1.0277`
- `R17` no-patch local probe isolated one exact short-prompt divergence:
  - `Llama-3.2-1B 256 no-group/current decode ratio ~1.10696x`
  - `Qwen2.5-1.5B 256 no-group/current decode ratio ~1.09965x`
  - token counts and token prefixes matched on both cases
- `R17` candidate is now rejected:
  - local `run_greedy`-only compare was positive
  - authoritative remote AC1 reruns did not yield a clean accepted-surface win
  - the stronger lesson is structural: the current accepted AC1 surface still runs through the benchmark helper, so a `run_greedy`-only slice is too narrow
- `R18` candidate is now rejected locally:
  - shared helper/core convergence remained directionally plausible
  - but the shared per-token generator loop regressed the weakest small-model short-prompt case badly enough that remote budget was not justified
- `R19` candidate is now rejected:
  - local evidence was strong on helper/core surfaces
  - the decisive authoritative remote weakest case still failed to improve
  - convergence-by-contract is no longer implementation-ready enough to justify another patch by inertia
- `R20` weakest-case helper-surface decomposition is complete:
  - current Llama-family implementation looks correct enough for this target
  - current Qwen-family focused sanity check also looks correct enough
  - the dominant measured gap on the weakest accepted surface is helper-path `schedule_next` / forward-build cost, not scalarization/materialization
- `R22` helper-surface resident token-lookahead candidate is now rejected:
  - local control/candidate compare stayed too small to justify promotion:
    - `Llama-3.2-1B 256 candidate/control decode ratio ~1.00480x`
    - `Llama-3.2-1B 2048 candidate/control decode ratio ~0.98952x`
  - authoritative remote control/candidate compare stayed slightly regressive in absolute MLXs decode throughput:
    - `Llama-3.2-1B 256 candidate/control decode ratio ~0.99904x`
    - `Llama-3.2-1B 2048 candidate/control decode ratio ~0.99686x`
  - the better candidate `mlxs/mlx_lm` ratios on the remote rerun were baseline drift, not a stronger MLXs result:
    - remote `mlx_lm 256` median decode moved `~191.75 -> ~191.12 tok/s`
    - remote `mlx_lm 2048` median decode moved `~149.50 -> ~143.42 tok/s`
- `R22` changes the ranking:
  - simplifying helper token choreography is not enough
  - the remaining weakest-case tax survives a near-`mlx_lm` resident token loop
  - the next leverage must move below the helper contract into the model/cache forward-step substrate
- `R23` framing is now recorded in a live audit artifact:
  - `docs/execution/milestones/M01-performance-architecture-redesign/R23_SUBSTRATE_AUDIT.md`
  - early no-patch readback narrows the next target to the prompt-tail / first-decode boundary:
    - simple steady-state cached forward stayed near parity (`MLXs ~0.01273s`, `mlx_lm ~0.01267s`)
    - simple first-decode-after-prefill read stayed directionally worse for MLXs (`~0.01480s` vs `~0.00989s`)
  - isolated synthetic first-call `KVCache.update_and_fetch()` asymmetry did not survive order reversal, so it is not yet patch-justifying evidence
- `R23` is now closed negatively after local + remote substrate probing:
  - current MLXs total prompt-tail / first-decode boundary was already better than comparable `mlx_lm`
  - the first no-patch integrated boundary probe was materially worse both locally and remotely
  - the decisive first-decode component is slower in isolation, but not in a way that produced a promotable bounded single-request redesign slice
- `R23` changes the ranking again:
  - the single-request prompt-tail / first-decode family is honestly exhausted for now
  - the next highest-leverage adjacent path is the batch-first redesign fallback
- `R24` rerank now reconstructs current batch truth from repaired evidence:
  - repaired direct comparator is the authoritative AC2 surface
  - old catastrophic post-`R10` AC2 beliefs are superseded
  - prompt `256` is the remaining open AC2 closure target, while prompt `2048` is already positive and must be preserved
- current batch rerank says the dominant remaining taxes are architectural, not micro:
  - resident prompt-to-generation handoff
  - extendable active batch ownership
  - cache substrate breadth for canonical residency
- `R24-S1` no-patch decomposition narrowed the first bounded slice further:
  - current MLXs fast-path activation is already competitive
  - the stable slower component is the first active-batch generation step
- `R24-S2` candidate is now rejected:
  - isolated first-step materialization improved strongly on the micro surface
  - but the real prompt-`256` scheduler target regressed on the approved remote host
  - the next evidence target is overlap across consecutive active-batch steps, not one-step materialization alone
- `R24-S3` is now classified decisively:
  - overlap/readiness is real
  - but not as a narrow standalone seam
  - the evidence points to a broader active-batch state-lifetime problem
- `R24-S4` first concrete candidate is already attempted, failed, and reverted:
  - rejected candidate: `R24-S4a resident dual-step _SharedFastBatch owner`
  - it must not remain the active path under the same thesis/name
- Outcome B chosen:
  - the batch family remains alive
  - the next active path is a genuinely new owner-boundary redesign, not a retry of `R24-S4a`
- `R25` isolated owner-boundary redesign is now rejected locally on the canonical prompt-`256` scheduler surface:
  - worktree: `/Users/federicofilippi/Desktop/MyProj/MLXs-m01-r25-canonical-batch-ownership`
  - token-resident extendable owner candidate:
    - requests/s `candidate/control ~0.79339x`
    - p50 TTFT `~1.06247x`
    - p95 completion `~1.31978x`
  - dual-ready extendable owner candidate:
    - requests/s `candidate/control ~0.73350x`
    - p50 TTFT `~1.02370x`
    - p95 completion `~1.36299x`
- `R25` changes the ranking again:
  - per-row-offset extendability must not be injected into the canonical aligned fast path by default
  - the repaired direct AC2 canonical compare is too static to govern late-admission redesign ranking by itself
- new staggered local direct compare now exposes the real dynamic batch-extension gap:
  - artifact: `/tmp/mlxs_m01_r25_staggered_local_20260420_1.json`
  - `mlxs/mlx_lm requests_per_s ~0.69428x`
  - exact output parity: `True`
  - this is the stronger next governing batch surface for extension work
- `R26` remote authoritative staggered baseline is now established:
  - artifact: `/tmp/mlxs_m01_r26_staggered_remote_20260420_1.json`
  - `mlxs/mlx_lm requests_per_s ~0.85088x`
  - exact output parity: `True`
- `R26` authoritative decomposition now isolates the decisive tax:
  - artifact: `/tmp/mlxs_m01_r26_staggered_decomp_remote_20260420_1.json`
  - MLXs late request first-token step `129` vs `mlx_lm 4`
  - MLXs late request completion step `255` vs `mlx_lm 131`
- `R26-S1` full dynamic late-admission owner is rejected locally:
  - requests/s `candidate/control ~0.77448x`
  - exact output parity: `False`
- `R26-S2` concurrent late prefill plus legacy continuation handoff is rejected locally:
  - requests/s `candidate/control ~0.99397x`
  - p50 TTFT `~0.94485x`
  - p95 completion `~1.04957x`
  - exact output parity: `True`
- `R26-S2` changes the ranking:
  - prompt-side admission semantics are the right direction
  - but synchronous full prompt work inside the main scheduler step is still too expensive
  - the next move must bound prompt-owner work per step rather than widen the static hot path
- `R26-S4` budgeted prompt-owner redesign is now rejected locally:
  - control artifact: `/tmp/mlxs_m01_r26s4_staggered_control_local_20260420_1.json`
  - candidate artifact: `/tmp/mlxs_m01_r26s4_staggered_candidate_local_20260420_1.json`
  - requests/s `candidate/control ~0.91001x`
  - p50 TTFT `~0.99142x`
  - p95 completion `~1.14951x`
  - exact output parity: `True`
- `R26-S4` decomposition clarifies the next ranking:
  - candidate decomposition artifact: `/tmp/mlxs_m01_r26s4b_staggered_decomp_candidate_local_20260420_1.json`
  - late request first-token step `4`
  - late request completion step `130`
  - remaining loss is no longer prompt-owner semantics; it is the cost of the current prompt-progress substrate itself
- `R26-S5` true-overlap prompt-progress substrate redesign is now rejected locally:
  - control artifact: `/tmp/mlxs_m01_r26s5_staggered_control_local_20260420_1.json`
  - candidate artifact: `/tmp/mlxs_m01_r26s5_staggered_candidate_local_20260420_1.json`
  - requests/s `candidate/control ~1.00040x`
  - p50 TTFT `~1.04423x`
  - p95 completion `~1.04108x`
  - exact output parity: `True`
- `R26-S5` decomposition confirms the family closure:
  - candidate decomposition artifact: `/tmp/mlxs_m01_r26s5_staggered_decomp_candidate_local_20260420_1.json`
  - late request first-token step `4`
  - late request completion step `130`
  - the family now fixes semantics but not throughput
- `R26` batch prompt-progress family is now closed for this phase:
  - no strong enough lever remains in this family on the current accepted surfaces
  - the next valid move is a global rerank
- `R27` global rerank is now complete:
  - `AC1` remains the strongest open target
  - the `AC1` front is already sufficiently understood
  - another mini-slice on `AC1` is no longer justified
- `R27` therefore chooses an integrated refactor as the next move:
  - `M01.R28 integrated AC1 canonical single-request progression contract refactor`
- `R28` integrated canonical progression contract refactor is now rejected locally:
  - control artifact: `/tmp/mlxs_m01_r28_control_local_20260420_2.json`
  - candidate artifact: `/tmp/mlxs_m01_r28_candidate_local_20260420_2.json`
  - `Llama-3.2-1B 256 decode ~0.98852x`, e2e `~1.01687x`
  - `Llama-3.2-1B 2048 decode ~0.99909x`, e2e `~1.00797x`
  - `Qwen2.5-1.5B 256 decode ~0.90711x`, e2e `~1.10514x`
  - `Qwen2.5-1.5B 2048 decode ~0.92558x`, e2e `~1.03484x`
  - local `Llama-3.2-3B 256` was unavailable, so remote was not justified after the local gate failed
- `R28` changes the ranking:
  - benchmark-local progression duplication is not the decisive remaining AC1 lever
  - the next move is no longer another contract-level refactor in the same family
  - the next move is a lower-level AC1 substrate rerank with explicit MLX capability reconsideration
- `R29` no-patch AC1 substrate rerank is now complete:
  - artifact: `/tmp/mlxs_m01_r29_substrate_local_20260420_1.json`
  - exact output parity `True` on all local accepted-sensitive probe cases
  - local decode ratios:
    - `Llama-3.2-1B 256 ~1.01389x`
    - `Llama-3.2-1B 2048 ~1.10801x`
    - `Qwen2.5-1.5B 256 ~1.02524x`
    - `Qwen2.5-1.5B 2048 ~0.97067x`
  - lower-boundary component deltas are mixed:
    - decode `model_forward_s` per token is worse for MLXs on `3/4` local cases
    - attention/cache/mask deltas are not stable enough across cases to define one dominant substrate bottleneck
    - no single lower-level internal or op-level component dominates across at least two accepted-sensitive cases strongly enough to justify a new AC1 implementation family
- `R29` changes the ranking:
  - current official MLX capabilities remain available:
    - local installed `mlx 0.31.1`
    - latest official MLX release still `v0.31.1`
    - streams, custom extensions, and custom Metal kernels are available
  - built-in MLX capability path is not justified yet for `AC1`
  - custom extension / custom Metal kernel path is not justified yet for `AC1`
  - the `AC1` front is now closed for this phase and the program reranks globally
- Reduced-run remote `R16` Class A reruns are truth-first guardrails only; they do not replace the accepted AC1 ledger.
- AC13 best fair prefill tuning stays below bar (`best eager prefill 1.02808x < 1.05x`).
- Deep research result is written down; `run_greedy` lookahead adoption is promoted; remaining blockers are architectural, not informational.
- The next redesign work must attack the strongest remaining active front without reopening closed `AC1` or `R26` families by inertia.
- `R30` local lower-boundary probe now narrows the batch readback further:
  - current accepted MLXs still pays architectural late-request starvation
  - below that, the measured lower-boundary split is eval-discipline / owner-pipeline cost rather than attention, mask, or cache-update kernels
  - standalone built-in-MLX and custom-kernel pivots are not justified on current evidence
- first integrated `R30` runtime candidate changes the ranking again:
  - parallel fast generation owners can repair late-request step positions
  - but they duplicate generation stepping (`256` fast-batch step calls vs upstream `133` generation-batch calls)
- the last acceptable `R30` candidate closes the family:
  - one-owner dynamic extension still missed the real throughput gate
  - the next valid path is broader than owner lifetime alone:
    - `R31 mixed-offset batch cache substrate redesign on the staggered surface`

## Accepted remote baseline

- Host: `llm@169.254.225.109`
- Accepted tree: `/tmp/mlxs_layer1_candidate_20260417_1`
- Authoritative interpreter: `/Users/Shared/mlx-cluster-venv/bin/python`

## Immediate next step

- Active next path:
  - `M01.R31 mixed-offset batch cache substrate redesign on the staggered surface`
- Chosen continuation:
  - `R30` is closed after its last acceptable attempt failed locally
  - the next valid path is broader than owner lifetime alone:
    - mixed-offset cache, mask, and rope substrate redesign on the staggered-only path
  - built-in MLX and custom-kernel pivots remain out unless a future probe isolates a true op-level bottleneck
- Supporting repo-native tooling added:
  - `benchmarks/mlxs_vs_mlx_lm/r30_ac2_substrate_probe.py`
  - `tests/unit/test_benchmarks/test_r30_ac2_substrate_probe.py`
