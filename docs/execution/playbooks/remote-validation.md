# Remote Validation Playbook

## Baseline

- Accepted remote host: `llm@169.254.225.109`
- Accepted remote tree: `/tmp/mlxs_layer1_candidate_20260417_1`
- Working interpreter: `/Users/Shared/mlx-cluster-venv/bin/python`
- Broken path to avoid: `/Users/llm/Desktop/MLXs/.venv/bin/python`

## Candidate workflow

1. Copy accepted remote tree to an isolated candidate path.
2. Sync only the bounded candidate diff.
3. Verify hashes or direct file content before running expensive slices.
4. Run focused remote smoke/benchmarks.
5. Promote only after local guardrails and candidate evidence hold.

## Promotion workflow

1. Apply minimal validated diff locally.
2. Sync same minimal diff to accepted remote tree.
3. Verify remote file hash/state.
4. Run serial accepted-remote rerun.

## Failure rule

- If remote state disagrees with local assumptions, stop the claim and verify the actual remote file tree before continuing.
