# Codex Guide

## Source of truth order

1. [STATUS.md](STATUS.md)
2. active milestone files under `milestones/`
3. [FREEZE.md](FREEZE.md)
4. [DECISIONS.md](DECISIONS.md)
5. [WORKLOG.md](WORKLOG.md)
6. governing repo docs under `docs/refactor/` and `docs/specs.md`

## File-backed rules

- No critical state only in chat.
- After meaningful progress, update the right file immediately.
- Keep files dense, operational, and link-heavy.
- Minimize duplication; `STATUS` is the summary, not the history.

## Benchmark discipline

- Local is for reading, editing, focused tests, smoke checks, local probes.
- Remote is authoritative for promotable performance/live claims.
- Serialize heavy remote benchmarks; do not run conflicting remote slices concurrently.
- Preserve accepted remote baseline; use isolated remote candidates for risky work.
- Keep artifact paths in `WORKLOG.md`.

## Validation order

1. focused local tests or smoke checks
2. local probe if needed
3. isolated remote validation for risky performance claims
4. accepted-remote rerun only for promotable claims

## Worktree policy

- Accepted local baseline stays free of speculative redesign patches.
- Use isolated worktrees for risky families.
- Promote only minimal validated diffs.
- Never reset or revert accepted dirty baseline files.

## Token policy

- No decorative prose.
- No repeated recap.
- Spend tokens on evidence, bottlenecks, patch rationale, regressions, acceptance closure.

## Reporting rule

For substantial sessions, use:

1. slice/session number
2. exact goal
3. governing evidence used
4. exact repo/spec state readback
5. evidence-gate decision
6. exact files changed
7. exact implementation notes
8. exact tests and remote probes/benchmarks run
9. exact result summary
10. whether the result is promotable
11. exact next slice/session or real stop condition
