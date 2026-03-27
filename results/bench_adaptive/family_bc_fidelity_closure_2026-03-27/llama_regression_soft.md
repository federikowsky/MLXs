# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 21.99 | 0.045471 | 128 | None |  | 0 | 0 |
| long_static | adaptive_full | 211.05 | 0.004738 | 128 | True | 132005888 | 0 | 0 |
| long_static | adaptive_soft | 180.23 | 0.005548 | 128 | True | 82231296 | 0 | 0 |
