# Skill: MLXs Bounded Integration Workflow

## Purpose
Use this skill when the next step is not a local optimization, but a bounded integration problem between existing layers or subsystems.

This is for controlled integration work, not for architecture redesign.

## Use this skill when
Use it when the task involves:
- Layer 3 to Layer 4 integration,
- engine-backed product/runtime exposure,
- scheduler/product-path integration,
- request lifecycle bridging,
- event/stream attachment,
- prompt-cache composition with runtime behavior,
- or similar "how do these pieces connect cleanly?" problems.

## What this skill is for
This skill exists to answer:
- what is the smallest valid integration slice,
- where should ownership live,
- what should remain unchanged,
- and how to prove the integration without redesigning the system.

## Inputs to inspect first
Before proposing changes, inspect:
- current code path,
- relevant protocol definitions,
- composition roots,
- relevant `docs/refactor/*`,
- current behavior probes,
- current lifecycle/stream/cancellation/state surfaces.

## Standard work loop
1. Identify the exact integration gap.
2. Confirm which layer currently owns what.
3. Confirm what the relevant docs/specs allow.
4. Write the smallest useful integration shape:
   - host/adaptor ownership,
   - request lifecycle mapping,
   - event mapping,
   - cancellation mapping,
   - fallback behavior,
   - and what remains out of scope.
5. Decide:
   - `Patch allowed`
   - or
   - `Continue investigation`
6. If patching:
   - implement the smallest integration slice that proves the shape,
   - validate without redesigning adjacent layers,
   - preserve semantics unless explicitly changing them.

## Boundaries
Integration work must not:
- collapse layer boundaries,
- move policy into the wrong layer,
- create god-objects,
- or use "temporary" shortcuts that become architecture drift.

## Output requirements
Always report:
1. exact integration gap
2. exact proposed bounded shape
3. host/adaptor ownership
4. request lifecycle mapping
5. event/stream mapping
6. cancellation/removal mapping
7. what remains explicitly out of scope
8. evidence-gate decision
9. exact next slice or stop condition

## Anti-patterns
Do not:
- treat integration as a license to redesign architecture,
- move layer ownership casually,
- or patch around a missing decision with hidden coupling.
