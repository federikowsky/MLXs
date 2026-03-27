# Adaptive KV benchmark summary

schema_version: 1

git_revision: 727ca24813a2c50d74e81338debc6bf780dc2efe


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 229.88 | 0.004350 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 231.60 | 0.004318 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 221.07 | 0.004524 | 64 | True | 44613632 | 0 | 0 |
