# Adaptive KV benchmark summary

schema_version: 1

git_revision: 83b88dbff34e5b07099eabf0db1dae8334920ad6


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 215.97 | 0.004630 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 229.54 | 0.004357 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 226.70 | 0.004411 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 220.04 | 0.004545 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 229.65 | 0.004354 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 225.71 | 0.004430 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 223.23 | 0.004480 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | non_adaptive | 233.20 | 0.004288 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 228.24 | 0.004381 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 221.12 | 0.004522 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 216.80 | 0.004613 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 212.44 | 0.004707 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 202.13 | 0.004947 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 219.75 | 0.004551 | 64 | True | 44613632 | 0 | 0 |
