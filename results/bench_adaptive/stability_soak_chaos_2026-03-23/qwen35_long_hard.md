# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 79.16 | 0.012632 | 128 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 78.47 | 0.012743 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_full | 79.62 | 0.012559 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 61.03 | 0.016385 | 128 | False | 47808512 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 61.32 | 0.016308 | 128 | False | 47808512 | 1 | 1 |
