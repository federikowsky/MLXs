# Adaptive KV benchmark summary

schema_version: 1

git_revision: cb3f751ebc45e834acf0b0739e1a3ce0bc87df75


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 115.13 | 0.008686 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 110.80 | 0.009025 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 87.29 | 0.011457 | 64 | True | 41058304 | 1 | 1 |
