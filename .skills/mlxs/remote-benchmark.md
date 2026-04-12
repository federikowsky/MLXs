# Skill: MLXs Remote Benchmark Execution

## Purpose
Use this skill whenever MLXs validation depends on the approved remote machine.
This skill standardizes remote benchmark execution, candidate/control hygiene, artifact capture, and clean-state discipline.

## Use this skill when
Use it when:
- remote benchmark results are decision-grade,
- the workstream requires remote validation,
- candidate vs control must be compared,
- benchmark identity/provenance must be trustworthy,
- or clean promoted state must be preserved.

## Required mindset
The remote machine is not just a place to "run something".
It is the authoritative validation surface for the relevant benchmark workflow.

Treat:
- repo identity,
- candidate/control cleanliness,
- benchmark artifact provenance,
- and restore/cleanup
as part of the benchmark itself.

## Standard preparation
Before benchmarking:
1. inspect local authoritative repo state,
2. inspect remote candidate repo state,
3. inspect remote control state if the workflow uses one,
4. verify exact HEAD / branch / dirty state,
5. verify whether the candidate is supposed to be clean or intentionally dirty,
6. verify the benchmark scripts and artifact destination.

## Candidate/control discipline
Maintain:
- local authoritative state,
- remote candidate state,
- remote control state,
with explicit identity.

Prefer explicit reporting of:
- HEAD,
- branch,
- dirty/clean status,
- fingerprint/hash where relevant,
- artifact path(s).

## Typical flow
1. Confirm clean approved baseline or approved candidate isolation.
2. Sync candidate changes to remote candidate path.
3. Leave remote control clean unless the workflow explicitly changes it.
4. Run environment/gate snapshot if part of the workflow.
5. Run the remote benchmark/probe.
6. Capture artifact path(s).
7. Read back metrics from the artifact(s).
8. Compare candidate vs control or candidate vs approved baseline.
9. Classify result.
10. If not promoted:
    - restore local and remote candidate cleanly.
11. If promoted:
    - checkpoint/tag/commit as appropriate.

## Canonical MLXs remote benchmark environment

Use these paths and commands as the default remote benchmark environment unless the active task explicitly overrides them.

### Local authoritative repo
- `/Users/federicofilippi/Desktop/MyProj/MLXs-core-exec`

### Official remote benchmark host
- `ssh llm@169.254.225.109`

### Remote candidate repo
- `/Users/llm/Desktop/MLXs`

### Remote control worktree
- `/tmp/mlxs_longcase_control`

### Canonical remote scripts
- gate snapshot:
  - `/Users/llm/bench/scripts/bench_gate.sh`
- long-case benchmark loop:
  - `/Users/llm/bench/scripts/run_longcase_loop.sh`

### Default remote discipline
Treat the local repo as the source of truth.
Treat the remote candidate repo as the execution target.
Treat the remote control worktree as the clean comparison surface when the workflow requires control-vs-candidate benchmarking.

Before any decision-grade benchmark:
1. inspect local identity state
2. inspect remote candidate identity state
3. inspect remote control identity state
4. report exact HEAD / branch / clean-dirty state
5. confirm the artifact destination path(s)

### Default artifact discipline
Write benchmark/probe artifacts to explicit `/tmp/...` paths on the remote host and report them exactly.
Do not summarize results without also reporting artifact paths.

### Default post-decision discipline
If a patch is not promoted:
- restore local and remote candidate to clean approved state,
- or explicitly stash/isolate the candidate and report that state.

If a patch is promoted:
- checkpoint/tag/commit as appropriate,
- and report the resulting approved identity state exactly.

## Artifact discipline
Every decision-grade remote run should capture:
- exact artifact path,
- source code identity,
- clean/dirty state if available,
- relevant measured metrics,
- and comparison basis.

## Output requirements
Always report:
1. exact remote commands used
2. exact artifact path(s)
3. exact code identity state
4. exact comparison numbers
5. classification
6. recommendation
7. post-decision cleanup/promotion state

## Anti-patterns
Do not:
- benchmark on a dirty state without saying so,
- compare against an unclear control,
- promote from ambiguous artifacts,
- or leave remote candidate polluted after a rejected patch.
