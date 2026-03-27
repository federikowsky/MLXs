# Adaptive KV benchmark summary

schema_version: 1

git_revision: cb3f751ebc45e834acf0b0739e1a3ce0bc87df75


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 83.57 | 0.011966 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 87.62 | 0.011413 | 64 | True | 45187072 | 0 | 0 |
| long_static | adaptive_soft | 83.92 | 0.011917 | 64 | True | 39092224 | 0 | 0 |
