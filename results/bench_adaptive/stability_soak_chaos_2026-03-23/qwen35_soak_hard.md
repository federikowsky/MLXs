# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 75.78 | 0.013195 | 128 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 76.01 | 0.013157 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_full | 75.40 | 0.013262 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_full | 75.17 | 0.013303 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_full | 76.17 | 0.013129 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 58.97 | 0.016959 | 128 | False | 47808512 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 58.99 | 0.016952 | 128 | False | 47808512 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 58.16 | 0.017194 | 128 | False | 47808512 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 58.92 | 0.016973 | 128 | False | 47808512 | 1 | 1 |
