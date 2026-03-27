# Adaptive KV benchmark summary

schema_version: 1

git_revision: afe6682864c9fdbd3bdbff04405eb33fd2a951e5


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 328.40 | 0.003045 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 322.97 | 0.003096 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 228.97 | 0.004367 | 64 | True | 71335936 | 6 | 1 |
