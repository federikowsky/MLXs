# Adaptive KV benchmark summary

schema_version: 1

git_revision: cb3f751ebc45e834acf0b0739e1a3ce0bc87df75


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 328.61 | 0.003043 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 320.88 | 0.003116 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 232.91 | 0.004293 | 64 | True | 71335936 | 6 | 1 |
