# Adaptive KV benchmark summary

schema_version: 1

git_revision: cb3f751ebc45e834acf0b0739e1a3ce0bc87df75


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 79.01 | 0.012657 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 82.79 | 0.012078 | 64 | True | 45187072 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 83.67 | 0.011952 | 64 | True | 38076416 | 0 | 0 |
