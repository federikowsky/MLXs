# Adaptive KV benchmark summary

schema_version: 1

git_revision: c74f2e725d7c7895371065b129fe119d86566764


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 299.67 | 0.003337 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 300.47 | 0.003328 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 304.80 | 0.003281 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 314.84 | 0.003176 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 225.83 | 0.004428 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 211.56 | 0.004727 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 221.24 | 0.004520 | 64 | True | 71335936 | 6 | 1 |
