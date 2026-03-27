# Adaptive KV benchmark summary

schema_version: 1

git_revision: ce093c90a631be804ce8b0401c58307ecebd0fca


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 248.38 | 0.004026 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 206.96 | 0.004832 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 65.93 | 0.015167 | 64 | True | 44613632 | 0 | 0 |
| topic_drift | non_adaptive | 225.65 | 0.004432 | 64 | None |  | 0 | 0 |
| topic_drift | adaptive_full | 200.59 | 0.004985 | 64 | True | 65945600 | 0 | 0 |
| topic_drift | adaptive_soft | 76.65 | 0.013046 | 64 | True | 44613632 | 0 | 0 |
| delayed_topic_return | non_adaptive | 229.27 | 0.004362 | 64 | None |  | 0 | 0 |
| delayed_topic_return | adaptive_full | 212.39 | 0.004708 | 64 | True | 65945600 | 0 | 0 |
| delayed_topic_return | adaptive_soft | 78.72 | 0.012704 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | non_adaptive | 235.60 | 0.004245 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 211.26 | 0.004733 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 77.40 | 0.012921 | 64 | True | 44613632 | 0 | 0 |
