# Autonomous Refactor Policy

Updated: 2026-04-20

## Status

Active and binding for MLXs runtime, performance, and redesign work.

## Purpose

This file exists to prevent future execution drift back into micro-family churn, mini-refactor defaults, and over-fragmented slice sequencing.

Use it together with:
- `AGENTS.md`
- `docs/execution/REDESIGN_MODE.md`

## Binding execution rules

- One main runtime/refactor front at a time.
- An optional secondary research or instrumentation front is allowed only if it directly supports the active main front.
- Small-step work is allowed only to locate and confirm a bottleneck.
- Once a front is sufficiently understood and has burned about 2-3 bounded families without a strong promotable win, the default next move must be exactly one of:
  - integrated refactor of the relevant layer or contract
  - honest closure of the front and a global rerank
- Long chains of `Rxx-S1/S2/S3/...` on the same local seam are forbidden by default.
- Such chains are allowed only if the file-backed system explicitly records why the front is still only in bottleneck-confirmation mode.

## Worktree / branch hygiene

- Every risky redesign path must use a dedicated worktree and branch.
- Each worktree maps to one active path only.
- When a new path opens, choose explicitly whether to reuse the current technical worktree or open a fresh one, and record that choice in the file-backed execution docs.
- When a path wins:
  - promote or commit correctly
  - update the file-backed system
  - clean up obsolete worktrees and branches by default
- When a path loses:
  - reject or revert cleanly
  - update the file-backed system
  - clean up the closed worktree and branch by default unless there is a specific recorded reason to retain them
- Do not allow stale worktrees or stale branches to accumulate.
- If a closed worktree or branch is retained, the file-backed system must say exactly why.

## Front classification rule

Before choosing the next technical move, answer explicitly:

1. Is this front still in bottleneck-confirmation mode?
2. Or is it already sufficiently understood that the next move must be integrated refactor or closure/rerank?

If the answer to the second question is yes:
- do not create another bounded family by default
- do not generate another local variant on the same seam by inertia

## Path selection rule

- Codex chooses the next path autonomously from file-backed truth.
- The user does not define slices by default once the direction is established.
- Specs and docs are informative governance inputs, not cages.
- If redesign changes the correct architecture, update the file-backed system after the result.

## Success / failure continuation rule

- On success:
  - promote or commit the winning state when appropriate
  - update the file-backed system
  - continue immediately into the next strongest path
- On failure:
  - reject or revert the losing state cleanly
  - update the file-backed system
  - continue immediately into the next strongest path

Do not stop for:
- ordinary checkpoints
- ordinary reranks
- family closure
- waiting for approval on a locally decidable next step

## Stop conditions

Stop only for:

1. a real architecture-boundary blocker with concrete evidence
2. a real strategic or model-policy fork not already governed by repo docs
3. a hard execution blocker
4. adjacent high-leverage paths honestly exhausted
5. remote-authoritative validation impossible exactly when needed

## MLX substrate rule

- Always consider whether newer built-in MLX functionality, custom extensions, or custom Metal kernels are relevant when the active front reaches a lower-level rerank or research phase.
- Do not treat that as permission to jump to kernels by default.
- First prove that the remaining bottleneck is truly substrate-level.
- Only then choose a built-in MLX capability path or a custom extension / custom Metal kernel path.
