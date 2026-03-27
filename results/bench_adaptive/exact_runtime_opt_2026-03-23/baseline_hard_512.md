# Adaptive KV benchmark summary

schema_version: 1

git_revision: 7666c6173e837801d66bc46e7a29d3c0ab147283


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 224.81 | 0.004448 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 212.12 | 0.004714 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 203.44 | 0.004915 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 207.53 | 0.004818 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 185.29 | 0.005397 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 184.18 | 0.005430 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 188.79 | 0.005297 | 64 | True | 41058304 | 1 | 1 |
