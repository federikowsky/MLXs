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
