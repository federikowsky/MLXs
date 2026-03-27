# Adaptive KV benchmark summary

schema_version: 1

git_revision: 83b88dbff34e5b07099eabf0db1dae8334920ad6


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 225.10 | 0.004442 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 230.33 | 0.004342 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 230.09 | 0.004346 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 230.73 | 0.004334 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 197.25 | 0.005070 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 202.19 | 0.004946 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 201.21 | 0.004970 | 64 | True | 41058304 | 1 | 1 |
