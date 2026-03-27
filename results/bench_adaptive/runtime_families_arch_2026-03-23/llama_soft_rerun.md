# Adaptive KV benchmark summary

schema_version: 1

git_revision: cb3f751ebc45e834acf0b0739e1a3ce0bc87df75


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 190.77 | 0.005242 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 153.80 | 0.006502 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 174.93 | 0.005716 | 64 | True | 44613632 | 0 | 0 |
