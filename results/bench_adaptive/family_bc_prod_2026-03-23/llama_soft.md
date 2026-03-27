# Adaptive KV benchmark summary

schema_version: 1

git_revision: cb3f751ebc45e834acf0b0739e1a3ce0bc87df75


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 164.09 | 0.006094 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 193.62 | 0.005165 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 191.49 | 0.005222 | 64 | True | 44613632 | 0 | 0 |
