# Skill: MLXs Performance Investigation and Decision Loop

## Purpose
Use this skill for MLXs runtime/performance work when the task is not just "change code", but:
- investigate,
- isolate,
- decide whether a patch is justified,
- benchmark correctly,
- and keep promoted state clean.

This skill is for evidence-first performance work, not casual optimization.

## Use this skill when
Use it when the task touches:
- core runtime hot paths,
- Layer 1 / Layer 2 / Layer 3 / Layer 4 performance,
- decode/prefill behavior,
- serving/orchestration performance,
- benchmark-driven promotion decisions,
- MLXs vs mlx_lm comparisons,
- or any performance-sensitive refactor.

## Inputs to inspect first
Before acting, inspect:
- current repository state,
- current clean/dirty state,
- current active branch / HEAD,
- relevant `docs/refactor/*`,
- baseline artifact(s),
- prior performance memos / dossiers / audits if present,
- benchmark scripts and current benchmark surface.

## Default work loop
1. Identify the exact performance surface.
2. Read the relevant code path.
3. Read the relevant `docs/refactor/*` governance docs.
4. Produce the minimum sufficient evidence artifact:
   - path walkthrough,
   - hot-path audit,
   - decomposition,
   - ranked targets,
   - or root-cause probe.
5. Make an evidence-gate decision:
   - `Patch allowed`
   - or
   - `Continue investigation`
6. If patch is allowed:
   - implement one bounded patch only,
   - validate locally as appropriate,
   - sync to the remote benchmark environment if required,
   - run decision-grade remote benchmarks,
   - classify result:
     - `Win`
     - `Neutral`
     - `Regression`
     - `Inconclusive`
7. Promote only if justified.
8. If not promoted:
   - restore clean approved state,
   - stash/revert/isolate candidate cleanly.
9. Reassess and continue only if no real stop condition exists.

## Evidence gate
A patch is allowed only if:
- the target is clearly identified,
- the problem is attributable,
- the expected improvement is meaningful,
- the validation surface is known,
- the slice is narrow,
- and the change does not require speculative redesign.

If not, continue investigation.

## Validation rules
Use the approved benchmark/validation path for the active workstream.
Do not treat ad hoc local timing as promotion-grade if the workstream relies on remote benchmark authority.

## Promotion rules
A patch may be promoted only if:
- result is clearly supported by approved validation,
- the candidate state is attributable and clean,
- and the new approved state is restored to a clean, checkpointable baseline.

## Required outputs
For each session, report:
1. session type
2. governing evidence used
3. verified facts
4. strong inferences
5. open questions
6. evidence-gate decision
7. if patching:
   - exact target
   - exact files changed
   - exact checks run
   - exact benchmark/probe artifacts
   - exact comparison
   - classification
   - recommendation
8. exact next step or stop condition

## Anti-patterns
Do not:
- patch just because something looks promising,
- optimize without a target,
- mix multiple optimizations in one patch,
- leave unpromoted patches contaminating approved state,
- or drift away from `docs/refactor/*` governance.
