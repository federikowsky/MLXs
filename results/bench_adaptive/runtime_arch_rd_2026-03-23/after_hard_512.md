# Adaptive KV benchmark summary

schema_version: 1

git_revision: 83b88dbff34e5b07099eabf0db1dae8334920ad6


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 202.74 | 0.004932 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 222.43 | 0.004496 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 218.96 | 0.004567 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 225.20 | 0.004440 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 205.53 | 0.004865 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 208.25 | 0.004802 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 198.40 | 0.005040 | 64 | True | 41058304 | 1 | 1 |
