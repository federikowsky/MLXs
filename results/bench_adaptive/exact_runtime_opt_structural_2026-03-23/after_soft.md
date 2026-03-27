# Adaptive KV benchmark summary

schema_version: 1

git_revision: c74f2e725d7c7895371065b129fe119d86566764


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 219.38 | 0.004558 | 64 | None |  | 0 | 0 |
| long_static | adaptive_full | 227.28 | 0.004400 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 236.84 | 0.004222 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_full | 234.01 | 0.004273 | 64 | True | 65945600 | 0 | 0 |
| long_static | adaptive_soft | 226.83 | 0.004409 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 227.10 | 0.004403 | 64 | True | 44613632 | 0 | 0 |
| long_static | adaptive_soft | 219.50 | 0.004556 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | non_adaptive | 236.62 | 0.004226 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 219.68 | 0.004552 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 204.35 | 0.004894 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 216.62 | 0.004616 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 217.04 | 0.004607 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 201.89 | 0.004953 | 64 | True | 44613632 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 223.62 | 0.004472 | 64 | True | 44613632 | 0 | 0 |
