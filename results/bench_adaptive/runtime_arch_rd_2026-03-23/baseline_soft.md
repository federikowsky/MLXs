# Adaptive KV benchmark summary

schema_version: 1

git_revision: 83b88dbff34e5b07099eabf0db1dae8334920ad6


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 234.60 | 0.004263 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 236.43 | 0.004230 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 235.11 | 0.004253 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 240.48 | 0.004158 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 236.23 | 0.004233 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 233.64 | 0.004280 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 229.29 | 0.004361 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | non_adaptive | 229.46 | 0.004358 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 219.75 | 0.004551 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 199.16 | 0.005021 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 215.02 | 0.004651 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 229.91 | 0.004349 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 214.70 | 0.004658 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 216.70 | 0.004615 | 64 | True | 44613632 | 0 | 0 |
