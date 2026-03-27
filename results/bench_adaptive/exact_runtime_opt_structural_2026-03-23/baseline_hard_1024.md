# Adaptive KV benchmark summary

schema_version: 1

git_revision: c74f2e725d7c7895371065b129fe119d86566764


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 269.63 | 0.003709 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 265.74 | 0.003763 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 263.96 | 0.003788 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 279.74 | 0.003575 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 197.24 | 0.005070 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 205.30 | 0.004871 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 202.89 | 0.004929 | 64 | True | 71335936 | 6 | 1 |
