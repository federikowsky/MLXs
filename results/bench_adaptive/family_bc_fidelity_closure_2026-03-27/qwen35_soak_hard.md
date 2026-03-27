# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 63.27 | 0.015806 | 128 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 61.91 | 0.016153 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_full | 56.96 | 0.017556 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_full | 44.72 | 0.022363 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_full | 53.08 | 0.018838 | 128 | True | 64061440 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 38.37 | 0.026062 | 128 | True | 47808512 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 45.91 | 0.021781 | 128 | True | 47808512 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 57.05 | 0.017527 | 128 | True | 47808512 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 47.71 | 0.020961 | 128 | True | 47808512 | 1 | 1 |
