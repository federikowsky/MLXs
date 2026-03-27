# Adaptive KV benchmark summary

schema_version: 1

git_revision: ce093c90a631be804ce8b0401c58307ecebd0fca


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| topic_drift | non_adaptive | 215.48 | 0.004641 | 64 | None |  | 0 | 0 |
| topic_drift | adaptive_full | 212.35 | 0.004709 | 64 | True | 65945600 | 0 | 0 |
| topic_drift | adaptive_soft | 67.51 | 0.014813 | 64 | True | 44613632 | 1 | 1 |
| topic_drift | adaptive_hard | 67.81 | 0.014748 | 64 | True | 41058304 | 1 | 1 |
| delayed_topic_return | non_adaptive | 222.08 | 0.004503 | 64 | None |  | 0 | 0 |
| delayed_topic_return | adaptive_full | 213.36 | 0.004687 | 64 | True | 65945600 | 0 | 0 |
| delayed_topic_return | adaptive_soft | 68.45 | 0.014609 | 64 | True | 44613632 | 1 | 1 |
| delayed_topic_return | adaptive_hard | 66.56 | 0.015025 | 64 | True | 41058304 | 1 | 1 |
| oscillation | non_adaptive | 228.29 | 0.004380 | 64 | None |  | 0 | 0 |
| oscillation | adaptive_full | 210.54 | 0.004750 | 64 | True | 65945600 | 0 | 0 |
| oscillation | adaptive_soft | 62.86 | 0.015910 | 64 | True | 44613632 | 1 | 1 |
| oscillation | adaptive_hard | 63.95 | 0.015637 | 64 | True | 41058304 | 1 | 1 |
| early_constraint_retention | non_adaptive | 213.44 | 0.004685 | 64 | None |  | 0 | 0 |
| early_constraint_retention | adaptive_full | 202.91 | 0.004928 | 64 | True | 65945600 | 0 | 0 |
| early_constraint_retention | adaptive_soft | 65.80 | 0.015198 | 64 | True | 44613632 | 1 | 1 |
| early_constraint_retention | adaptive_hard | 49.42 | 0.020235 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | non_adaptive | 211.01 | 0.004739 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 196.82 | 0.005081 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 66.93 | 0.014940 | 64 | True | 44613632 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 64.03 | 0.015619 | 64 | True | 41058304 | 1 | 1 |
