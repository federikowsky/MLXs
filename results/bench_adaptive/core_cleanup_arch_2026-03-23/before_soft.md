# Adaptive KV benchmark summary

schema_version: 1

git_revision: 727ca24813a2c50d74e81338debc6bf780dc2efe


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 78.48 | 0.012742 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 80.43 | 0.012433 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 67.73 | 0.014764 | 64 | True | 44613632 | 0 | 0 |
