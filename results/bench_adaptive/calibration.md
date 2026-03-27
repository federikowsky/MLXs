# Adaptive KV benchmark summary

schema_version: 1

git_revision: ce093c90a631be804ce8b0401c58307ecebd0fca


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| smoke | non_adaptive | 48.03 | 0.020818 | 8 | None |  | 0 | 0 |
| smoke | adaptive_full | 135.99 | 0.007353 | 8 | True | 4472832 | 0 | 0 |
| long_static | non_adaptive | 223.06 | 0.004483 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 200.19 | 0.004995 | 64 | True | 65945600 | 0 | 0 |
