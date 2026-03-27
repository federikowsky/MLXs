# Adaptive KV benchmark summary

schema_version: 1

git_revision: 727ca24813a2c50d74e81338debc6bf780dc2efe


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 224.19 | 0.004461 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 228.38 | 0.004379 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 224.79 | 0.004449 | 64 | True | 44613632 | 0 | 0 |
