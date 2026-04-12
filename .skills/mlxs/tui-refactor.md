# Skill: MLXs Chat TUI Refactor Workflow

## Purpose
Use this skill for the MLXs chat TUI / product-surface refactor workstream.

This skill is for disciplined UI foundation work:
- splitting ownership cleanly,
- preserving behavior while refactoring,
- and moving toward a modern transcript-first professional TUI.

## Use this skill when
Use it when the task touches:
- `chat/present/*`
- `chat/tui/*`
- `chat/cli.py`
- `chat/store.py`
- `product_surfaces/chat.py`
- shell/layout/scaffold/presentation extraction
- bounded Phase 2 scaffold work

## Product bar
The target is:
- transcript-first
- calm
- professional
- modern
- clean
- high-readability
- strong hierarchy
- excellent DX/UX

The target is NOT:
- a demo,
- an MVP,
- a nostalgic terminal dashboard,
- a noisy sysadmin tool,
- or a cluttered panel wall.

## Architectural ownership
Respect these boundaries:
- `chat/present/` owns pure presentation and view-model shaping
- `chat/tui/` owns prompt_toolkit mechanics, layout, widgets, focus, scaffold
- `product_surfaces/chat.py` remains controller/orchestration
- `chat/store.py` remains minimal product-state
- `chat/session.py` remains canonical message/session model
- `chat/input.py` remains canonical resolution path

## Standard work loop
1. Inspect current repo state.
2. Determine current phase:
   - foundation / refactor,
   - scaffold,
   - or later UX work.
3. Decide whether the next correct move is:
   - one more bounded extraction,
   - one small scaffold slice,
   - or a real phase/design stop.
4. Make one bounded slice only.
5. Validate.
6. Reassess and continue autonomously if justified.

## Preferred Phase 1 behavior
For foundation work:
- prefer behavior-preserving slices,
- extract pure modules first,
- shrink `chat/cli.py`,
- keep the store minimal,
- do not expand UX prematurely.

## Preferred Phase 2 behavior
For scaffold work:
- begin with layout/scaffold foundations only,
- keep transcript dominant,
- keep persistent chrome restrained,
- avoid noisy expansions,
- do not jump straight into the full workstation UI.

## Output requirements
For each slice:
1. exact goal
2. exact files changed
3. exact implementation notes
4. exact tests added/updated
5. exact commands run
6. result summary
7. ready-to-commit decision
8. recommended commit message
9. exact next slice

## Anti-patterns
Do not:
- mix foundation refactor with large UX expansion,
- let widgets absorb business logic,
- let the store become persistence/runtime logic,
- dump help/errors/system noise into the transcript when cleaner UI surfaces are intended,
- or drift toward dashboard-wall aesthetics.
