# Adaptive KV benchmark summary

schema_version: 1

git_revision: cb3f751ebc45e834acf0b0739e1a3ce0bc87df75


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 205.20 | 0.004873 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 208.44 | 0.004798 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 187.98 | 0.005320 | 64 | True | 44613632 | 0 | 0 |
