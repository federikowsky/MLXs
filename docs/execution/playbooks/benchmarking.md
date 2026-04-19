# Benchmarking Playbook

## Default policy

- Use local for focused probes.
- Use remote for promotable performance claims.
- Use accepted remote baseline unless running an isolated candidate.

## Remote authority

- Host: `llm@169.254.225.109`
- Accepted tree: `/tmp/mlxs_layer1_candidate_20260417_1`
- Interpreter: `/Users/Shared/mlx-cluster-venv/bin/python`

## Rules

- Serialize heavy remote benchmarks.
- Preserve shared prompt-token equivalence.
- Record exact artifact paths.
- Compare new artifacts directly against accepted artifacts, not memory.
- Do not make promotable claims from local-only evidence.

## Artifact naming

- Local probe: `/tmp/mlxs_<family>_local_<date>_<n>.json`
- Remote candidate: `/tmp/mlxs_<family>_candidate_remote_<date>_<n>.json`
- Accepted remote rerun: `/tmp/mlxs_<family>_promoted_remote_<date>_<n>.json`
- Ranked compare: `/tmp/mlxs_<family>_ranked_<date>_<n>.json`
