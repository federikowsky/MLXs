# Adaptive KV benchmark summary

schema_version: 1

git_revision: ce093c90a631be804ce8b0401c58307ecebd0fca


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 227.93 | 0.004387 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 213.98 | 0.004673 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_hard | 66.12 | 0.015123 | 64 | True | 41058304 | 1 | 1 |
| topic_drift | non_adaptive | 225.91 | 0.004427 | 64 | None |  | 0 | 0 |
| topic_drift | adaptive_full | 207.20 | 0.004826 | 64 | True | 65945600 | 0 | 0 |
| topic_drift | adaptive_hard | 68.26 | 0.014649 | 64 | True | 41058304 | 1 | 1 |
| delayed_topic_return | non_adaptive | 229.90 | 0.004350 | 64 | None |  | 0 | 0 |
| delayed_topic_return | adaptive_full | 213.36 | 0.004687 | 64 | True | 65945600 | 0 | 0 |
| delayed_topic_return | adaptive_hard | 65.19 | 0.015341 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | non_adaptive | 224.43 | 0.004456 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 213.12 | 0.004692 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 67.75 | 0.014761 | 64 | True | 41058304 | 1 | 1 |
