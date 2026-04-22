# M01.R28 AC1 Canonical Progress Contract Refactor

Updated: 2026-04-20

## Why this path exists

`R27` closes the `R26` batch prompt-progress family and reranks the remaining open fronts.

Current result:

- `AC1` remains the strongest open target
- the `AC1` front is already sufficiently understood
- it has already exhausted multiple bounded families without a strong promotable win:
  - `R17`
  - `R18`
  - `R19`
  - `R22`
  - `R23`

Under the project operating rule, another micro-slice is no longer justified here.

## Main thesis

- `R28 integrated AC1 canonical single-request progression contract refactor`

Core idea:

- stop treating the benchmark helper and canonical Layer 1 runtime as separate progression owners
- introduce one canonical single-request progression contract in Layer 1
- make both `runtime_core.run_greedy` and the Class A benchmark path consume that same contract
- move benchmark-specific lookahead policy down to a governed Layer 1 contract boundary instead of letting the benchmark own its own progression loop

## Fallback thesis

If the full contract refactor is too broad as a first move, first collapse the benchmark/helper progression logic behind one Layer 1-owned iterator or driver while leaving the public benchmark surface unchanged.

## Do not pursue

- no more helper/core convergence micro-slices under the old `R17`/`R18`/`R19` family
- no more helper-only token choreography redesigns under the old `R22` family
- no more benchmark-local progression loops as the default long-term architecture
