# Adaptive KV benchmark summary

schema_version: 1

git_revision: 83b88dbff34e5b07099eabf0db1dae8334920ad6


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 284.88 | 0.003510 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 289.66 | 0.003452 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 313.02 | 0.003195 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 315.60 | 0.003169 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 214.02 | 0.004673 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 195.93 | 0.005104 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 199.08 | 0.005023 | 64 | True | 71335936 | 6 | 1 |
