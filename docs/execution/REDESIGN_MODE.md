# Redesign Mode

Updated: 2026-04-20

## Status

Active and binding for MLXs performance redesign work.

## Directive

- No more micro-fix, micro-variant, or micro-slice churn as the default operating pattern.
- No more mini-family or mini-refactor default behavior on already-understood fronts.
- Work objective-first and autonomy-first against the strongest open architectural leverage.
- Codex is the main architect, planner, and executor for this redesign program unless a true hard blocker is reached.
- The file-backed execution system under `docs/execution/` is the source of truth; do not rely on chat memory as authority.
- The strengthened runtime/refactor execution policy in `docs/execution/REFACTOR_POLICY.md` is binding and must be followed for front selection and continuation.
- Larger integrated refactors are allowed when the evidence says they are the highest-leverage path.
- Existing specs and docs are informative governance inputs, not cages; when redesign changes the truth, update the file-backed system to match the new truth.
- If one path closes negatively, continue immediately into the next strongest ranked path instead of stalling on process churn.
- Small-step work is allowed only to locate and confirm a bottleneck.
- If a front is already well understood and has burned about 2-3 bounded families without a strong promotable win, the default next move must be exactly one of:
  - integrated refactor of the relevant layer or contract
  - honest closure and rerank
- Do not generate long `Rxx-S1/S2/S3/...` chains on the same local seam unless the file-backed system explicitly records why the front is still only in bottleneck-confirmation mode.
- Keep one main runtime/refactor front active at a time.
- A secondary research or instrumentation front is allowed only when it directly supports the main front.
- Codex chooses the next path autonomously from file-backed truth.
- The user no longer defines slices by default once the direction is established.
- On success: promote or commit the winning state, update file-backed docs, and continue immediately.
- On failure: reject or revert cleanly, update file-backed docs, and continue immediately.
- Consider built-in MLX capabilities, custom extensions, and custom Metal kernels when relevant, but only after the bottleneck is proven to be substrate-level.
- Stop only for true hard blockers:
  - architecture or strategy forks not already governed by repo docs,
  - missing required information that cannot be recovered from repo state or approved tools,
  - execution or environment blockers that prevent safe continuation,
  - remote-authoritative validation becoming impossible exactly when promotion requires it,
  - adjacent high-leverage redesign paths being honestly exhausted.

## Operating expectations

1. Re-ground from current accepted evidence before choosing the next redesign path.
2. Produce a ranked redesign thesis, a fallback thesis, and an explicit do-not-pursue list.
3. Explicitly classify whether the current front is still in bottleneck-confirmation mode or has already crossed into integrated-refactor-or-close mode.
4. Choose the implementation shape that best tests the thesis, even when that means a multi-file integrated refactor.
5. Keep risky redesign code inside isolated worktrees until it wins.
6. Validate in order:
   - local correctness
   - local benchmark truth
   - remote authoritative validation for promotable claims
7. After every meaningful result, update the file-backed execution artifacts before moving on.

## Non-negotiables

- Aggressive redesign is allowed.
- Sloppy redesign is not.
- Benchmark truth is mandatory.
- Accepted baseline safety is mandatory.
- Promotion requires evidence, not intuition.
