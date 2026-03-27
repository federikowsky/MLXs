# Adaptive KV benchmark summary

schema_version: 1

git_revision: 727ca24813a2c50d74e81338debc6bf780dc2efe


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 78.06 | 0.012810 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 78.57 | 0.012728 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 68.78 | 0.014539 | 64 | True | 41058304 | 1 | 1 |
