# Adaptive KV benchmark summary

schema_version: 1

git_revision: afe6682864c9fdbd3bdbff04405eb33fd2a951e5


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 217.71 | 0.004593 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 246.44 | 0.004058 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 244.28 | 0.004094 | 64 | True | 44613632 | 0 | 0 |
