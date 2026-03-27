# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 195.74 | 0.005109 | 128 | None |  | 0 | 0 |
| long_static | adaptive_full | 188.95 | 0.005292 | 128 | True | 132005888 | 0 | 0 |
| long_static | adaptive_full | 188.99 | 0.005291 | 128 | True | 132005888 | 0 | 0 |
| long_static | adaptive_full | 188.22 | 0.005313 | 128 | True | 132005888 | 0 | 0 |
| long_static | adaptive_full | 189.16 | 0.005286 | 128 | True | 132005888 | 0 | 0 |
| long_static | adaptive_soft | 180.00 | 0.005555 | 128 | True | 82231296 | 0 | 0 |
| long_static | adaptive_soft | 179.30 | 0.005577 | 128 | True | 82231296 | 0 | 0 |
| long_static | adaptive_soft | 176.77 | 0.005657 | 128 | True | 82231296 | 0 | 0 |
| long_static | adaptive_soft | 177.38 | 0.005638 | 128 | True | 82231296 | 0 | 0 |
