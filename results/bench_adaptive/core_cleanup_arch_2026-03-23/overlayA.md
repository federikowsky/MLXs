# Adaptive KV benchmark summary

schema_version: 1

git_revision: 727ca24813a2c50d74e81338debc6bf780dc2efe


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 313.00 | 0.003195 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 307.93 | 0.003247 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 227.19 | 0.004402 | 64 | True | 71335936 | 6 | 1 |
