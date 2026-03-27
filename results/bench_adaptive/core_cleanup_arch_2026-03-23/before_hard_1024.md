# Adaptive KV benchmark summary

schema_version: 1

git_revision: 727ca24813a2c50d74e81338debc6bf780dc2efe


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 93.02 | 0.010751 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 116.31 | 0.008598 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 162.53 | 0.006153 | 64 | True | 71335936 | 6 | 1 |
