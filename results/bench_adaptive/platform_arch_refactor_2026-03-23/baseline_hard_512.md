# Adaptive KV benchmark summary

schema_version: 1

git_revision: afe6682864c9fdbd3bdbff04405eb33fd2a951e5


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 238.32 | 0.004196 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 246.32 | 0.004060 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 226.95 | 0.004406 | 64 | True | 41058304 | 1 | 1 |
