# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 100.64 | 0.009936 | 128 | None |  | 0 | 0 |
| long_static | adaptive_full | 51.38 | 0.019462 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_full | 50.18 | 0.019929 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_full | 56.78 | 0.017613 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_full | 46.95 | 0.021297 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_soft | 53.66 | 0.018637 | 128 | False | 48824320 | 0 | 0 |
| long_static | adaptive_soft | 50.89 | 0.019649 | 128 | False | 48824320 | 0 | 0 |
| long_static | adaptive_soft | 56.64 | 0.017656 | 128 | False | 48824320 | 0 | 0 |
| long_static | adaptive_soft | 56.47 | 0.017708 | 128 | False | 48824320 | 0 | 0 |
