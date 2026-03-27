# Adaptive KV benchmark summary

schema_version: 1

git_revision: afe6682864c9fdbd3bdbff04405eb33fd2a951e5


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 235.88 | 0.004239 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 221.49 | 0.004515 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 214.78 | 0.004656 | 64 | True | 41058304 | 1 | 1 |
