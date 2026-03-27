# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 127.11 | 0.007867 | 256 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 123.83 | 0.008076 | 256 | True | 146685952 | 0 | 0 |
| hard_pressure_context | adaptive_full | 127.02 | 0.007873 | 256 | True | 146685952 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 104.75 | 0.009546 | 256 | True | 82690048 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 105.09 | 0.009516 | 256 | True | 82690048 | 6 | 1 |
