# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 14.52 | 0.068894 | 128 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 38.60 | 0.025905 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 13.43 | 0.074453 | 128 | True | 47808512 | 1 | 1 |
