# Adaptive KV benchmark summary

schema_version: 1

git_revision: 7666c6173e837801d66bc46e7a29d3c0ab147283


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 223.34 | 0.004477 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 221.03 | 0.004524 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 232.99 | 0.004292 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 233.49 | 0.004283 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 206.42 | 0.004844 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 210.49 | 0.004751 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 211.22 | 0.004734 | 64 | True | 41058304 | 1 | 1 |
