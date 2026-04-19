# Execution Index

This directory is the file-backed operating system of record.

## Truth locations

- Current real state: [STATUS.md](STATUS.md)
- Macro sequencing and milestone state: [ROADMAP.md](ROADMAP.md)
- Active milestone plan/checklist/risks/report:
  - [milestones/M01-performance-architecture-redesign/PLAN.md](milestones/M01-performance-architecture-redesign/PLAN.md)
  - [milestones/M01-performance-architecture-redesign/TASKS.md](milestones/M01-performance-architecture-redesign/TASKS.md)
  - [milestones/M01-performance-architecture-redesign/RISKS.md](milestones/M01-performance-architecture-redesign/RISKS.md)
  - [milestones/M01-performance-architecture-redesign/REPORT.md](milestones/M01-performance-architecture-redesign/REPORT.md)
- Accepted/rejected/superseded decisions: [DECISIONS.md](DECISIONS.md)
- Execution history and artifact ledger: [WORKLOG.md](WORKLOG.md)
- Frozen seams, accepted baselines, blocked families: [FREEZE.md](FREEZE.md)
- Live deferred work: [BACKLOG.md](BACKLOG.md)
- Agent operating rules for this repo: [CODEX_GUIDE.md](CODEX_GUIDE.md)
- Critical repo surface map: [REPO_MAP.md](REPO_MAP.md)
- Recurring protocols:
  - [playbooks/benchmarking.md](playbooks/benchmarking.md)
  - [playbooks/remote-validation.md](playbooks/remote-validation.md)
  - [playbooks/worktree-policy.md](playbooks/worktree-policy.md)
  - [playbooks/file-backed-ops.md](playbooks/file-backed-ops.md)

## Update order

1. Update [STATUS.md](STATUS.md) when active state changes.
2. Update milestone [TASKS.md](milestones/M01-performance-architecture-redesign/TASKS.md) and [RISKS.md](milestones/M01-performance-architecture-redesign/RISKS.md) when work advances.
3. Append durable choices to [DECISIONS.md](DECISIONS.md).
4. Append slice history and artifact paths to [WORKLOG.md](WORKLOG.md).
5. Update [FREEZE.md](FREEZE.md) only when a family is materially closed or blocked.

## Scope

- Critical state must not live only in chat.
- Keep files dense and operational.
- Prefer links over duplication.
