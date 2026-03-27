# Adaptive KV benchmark summary

schema_version: 1

git_revision: 83b88dbff34e5b07099eabf0db1dae8334920ad6


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 274.19 | 0.003647 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 292.12 | 0.003423 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 194.63 | 0.005138 | 64 | True | 71335936 | 6 | 1 |
