# M01 Report

Milestone not closed.

## Durable checkpoint

- Commit: `fcb69de`
- Type: milestone checkpoint commit
- Captures:
  - accepted promoted baseline through `R10`
  - promoted shared Layer 1 + Layer 3 aligned plain-KV batch progression kernel
  - promoted scheduler-level AC2 probe tooling
  - file-backed execution state through active handoff `R11`

## Current promoted fronts

- `M01.5` benchmark-helper lookahead branch
- `M01.R2` `runtime_core.run_greedy` lookahead adoption
- `M01.R10` shared Layer 1 + Layer 3 aligned plain-KV batch progression kernel

## Rejected redesign families

- class-based resident progression container
- data-only resident progression container
- broad inline helper migration
- broad and narrowed `general_path` helper migrations
- `R6` resident completion-batch object
- `R7` aligned cohort prefill-to-completion handoff
- `R9` private Layer 3-only resident fast-batch contract

## Carry-forward

- milestone remains open
- active next path: `M01.R11 post-R10 acceptance closure rerank`
