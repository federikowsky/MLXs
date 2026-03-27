# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 14.02 | 0.071340 | 128 | None |  | 0 | 0 |
| long_static | adaptive_full | 41.48 | 0.024106 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_soft | 13.56 | 0.073771 | 128 | False | 48824320 | 0 | 0 |
