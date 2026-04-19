# Roadmap

## State key

- `active`
- `completed`
- `pending`
- `blocked`

## Milestones

| Milestone | State | Depends on | Goal |
| --- | --- | --- | --- |
| `M00-baseline-hardening-and-parity-cleanup` | `completed` | — | Recover credible baseline, close bounded compatibility and benchmark-surface wins, establish accepted remote artifacts. |
| `M01-performance-architecture-redesign` | `active` | `M00` | Replace exhausted micro-fix loops with structural hot-path redesign aimed at AC1/AC2 leverage. |
| `M02-acceptance-closure` | `pending` | `M01` | Re-run authoritative acceptance targets after redesign and close or honestly bound AC1/AC2/AC13. |
| `M03-product-ready-closeout` | `pending` | `M02` | Finish remaining product-ready acceptance items and stabilize frozen architecture. |

## M01 dependency framing

- Needs the accepted promoted baseline from `M00`.
- Must preserve accepted Qwen/Llama benchmarkability and product-surface fixes.
- Must use remote-authoritative benchmarking for promotable performance claims.

## Exit criteria for M01

- First redesign front chosen from evidence, not intuition.
- First redesign worktree opened only after front ranking.
- At least one architecture-level candidate validated against AC1/AC2 surfaces.
- Post-redesign acceptance sequencing prepared for `M02`.
