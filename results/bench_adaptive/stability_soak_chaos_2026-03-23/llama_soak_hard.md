# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 189.96 | 0.005264 | 128 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 185.63 | 0.005387 | 128 | True | 132005888 | 0 | 0 |
| hard_pressure_context | adaptive_full | 185.49 | 0.005391 | 128 | True | 132005888 | 0 | 0 |
| hard_pressure_context | adaptive_full | 184.62 | 0.005416 | 128 | True | 132005888 | 0 | 0 |
| hard_pressure_context | adaptive_full | 182.19 | 0.005489 | 128 | True | 132005888 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 140.15 | 0.007135 | 128 | True | 75120640 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 138.22 | 0.007235 | 128 | True | 75120640 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 141.05 | 0.007090 | 128 | True | 75120640 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 141.13 | 0.007086 | 128 | True | 75120640 | 6 | 1 |
