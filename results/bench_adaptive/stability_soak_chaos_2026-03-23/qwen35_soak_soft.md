# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 65.77 | 0.015203 | 128 | None |  | 0 | 0 |
| long_static | adaptive_full | 74.86 | 0.013358 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_full | 75.28 | 0.013284 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_full | 76.06 | 0.013147 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_full | 74.95 | 0.013342 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_soft | 76.02 | 0.013154 | 128 | False | 48824320 | 0 | 0 |
| long_static | adaptive_soft | 75.92 | 0.013171 | 128 | False | 48824320 | 0 | 0 |
| long_static | adaptive_soft | 76.59 | 0.013056 | 128 | False | 48824320 | 0 | 0 |
| long_static | adaptive_soft | 76.21 | 0.013121 | 128 | False | 48824320 | 0 | 0 |
