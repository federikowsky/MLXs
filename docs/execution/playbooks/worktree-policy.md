# Worktree Policy

## Use worktrees for

- risky performance families
- compatibility families
- redesign candidates
- any bounded experiment that should not touch the accepted baseline

## Naming

- local worktree: `MLXs-<family>`
- branch: `codex/<family>`

## Rules

- Do not patch the accepted baseline first for risky work.
- Keep losing candidates inside their isolated worktree only.
- Promote only the minimal validated diff.
- Do not reset or revert accepted dirty files.
- Benchmark-surface-only fixes are acceptable if they are explicit and bounded.
