# Adaptive KV benchmark summary

schema_version: 1

git_revision: 7666c6173e837801d66bc46e7a29d3c0ab147283


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 225.65 | 0.004432 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 228.18 | 0.004383 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 233.71 | 0.004279 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 229.70 | 0.004354 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 216.15 | 0.004626 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 225.82 | 0.004428 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 221.94 | 0.004506 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | non_adaptive | 241.24 | 0.004145 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 235.74 | 0.004242 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 233.66 | 0.004280 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 233.62 | 0.004280 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 231.05 | 0.004328 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 225.46 | 0.004435 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 225.08 | 0.004443 | 64 | True | 44613632 | 0 | 0 |
