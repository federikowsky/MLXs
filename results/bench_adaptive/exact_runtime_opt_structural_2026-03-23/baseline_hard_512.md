# Adaptive KV benchmark summary

schema_version: 1

git_revision: c74f2e725d7c7895371065b129fe119d86566764


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 201.20 | 0.004970 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 205.69 | 0.004862 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 201.49 | 0.004963 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 212.55 | 0.004705 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 186.54 | 0.005361 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 188.25 | 0.005312 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 187.84 | 0.005324 | 64 | True | 41058304 | 1 | 1 |
