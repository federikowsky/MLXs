# Adaptive KV benchmark summary

schema_version: 1

git_revision: c74f2e725d7c7895371065b129fe119d86566764


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 209.86 | 0.004765 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 195.72 | 0.005109 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 194.97 | 0.005129 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 196.15 | 0.005098 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 201.21 | 0.004970 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 207.82 | 0.004812 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 176.69 | 0.005660 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | non_adaptive | 211.24 | 0.004734 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 206.70 | 0.004838 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 199.76 | 0.005006 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 203.96 | 0.004903 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 201.25 | 0.004969 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 197.51 | 0.005063 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 198.57 | 0.005036 | 64 | True | 44613632 | 0 | 0 |
