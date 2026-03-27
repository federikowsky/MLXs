# Adaptive KV benchmark summary

schema_version: 1

git_revision: afe6682864c9fdbd3bdbff04405eb33fd2a951e5


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 236.44 | 0.004229 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 249.04 | 0.004015 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 240.11 | 0.004165 | 64 | True | 44613632 | 0 | 0 |
