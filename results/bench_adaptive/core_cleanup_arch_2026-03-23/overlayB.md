# Adaptive KV benchmark summary

schema_version: 1

git_revision: 727ca24813a2c50d74e81338debc6bf780dc2efe


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 310.81 | 0.003217 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 318.31 | 0.003142 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 230.90 | 0.004331 | 64 | True | 71335936 | 6 | 1 |
