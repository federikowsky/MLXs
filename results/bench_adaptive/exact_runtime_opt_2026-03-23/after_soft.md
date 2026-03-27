# Adaptive KV benchmark summary

schema_version: 1

git_revision: 7666c6173e837801d66bc46e7a29d3c0ab147283


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 227.07 | 0.004404 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 232.13 | 0.004308 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 224.44 | 0.004456 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 234.89 | 0.004257 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 234.53 | 0.004264 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 225.56 | 0.004433 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 223.19 | 0.004480 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | non_adaptive | 237.68 | 0.004207 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 235.02 | 0.004255 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 234.30 | 0.004268 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 233.80 | 0.004277 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 226.89 | 0.004407 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 227.22 | 0.004401 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 219.33 | 0.004559 | 64 | True | 44613632 | 0 | 0 |
