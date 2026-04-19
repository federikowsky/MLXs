# File-Backed Ops Playbook

## Mandatory updates

- `STATUS.md`: active milestone, workstream, blockers, immediate next step
- milestone `TASKS.md`: active checklist state
- milestone `RISKS.md`: open risks/blockers
- `DECISIONS.md`: durable accepted/rejected choices
- `WORKLOG.md`: artifacts, worktrees, useful session history
- `FREEZE.md`: only when a family materially closes or blocks

## Writing rules

- Keep entries dense.
- Prefer evidence paths over prose.
- Do not mirror the same full state in multiple files.
- Critical chat-only state must be copied into the right file before moving on.

## Session close rule

Before ending a substantial slice:

1. update the milestone files
2. update `STATUS.md`
3. append `WORKLOG.md`
4. append `DECISIONS.md` if a real decision happened
5. update `FREEZE.md` only if a seam materially closed
