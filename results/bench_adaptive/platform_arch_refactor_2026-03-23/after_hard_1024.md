# Adaptive KV benchmark summary

schema_version: 1

git_revision: afe6682864c9fdbd3bdbff04405eb33fd2a951e5


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 321.36 | 0.003112 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 319.87 | 0.003126 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 231.10 | 0.004327 | 64 | True | 71335936 | 6 | 1 |
