# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 81.35 | 0.012293 | 128 | None |  | 0 | 0 |
| long_static | adaptive_full | 81.78 | 0.012227 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_full | 77.37 | 0.012925 | 128 | True | 64061440 | 0 | 0 |
| long_static | adaptive_soft | 74.16 | 0.013485 | 128 | False | 48824320 | 0 | 0 |
| long_static | adaptive_soft | 80.02 | 0.012497 | 128 | False | 48824320 | 0 | 0 |
