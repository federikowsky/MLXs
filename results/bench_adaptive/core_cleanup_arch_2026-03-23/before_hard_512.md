# Adaptive KV benchmark summary

schema_version: 1

git_revision: 727ca24813a2c50d74e81338debc6bf780dc2efe


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 80.95 | 0.012354 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 76.15 | 0.013133 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 63.77 | 0.015681 | 64 | True | 41058304 | 1 | 1 |
