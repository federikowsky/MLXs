# Adaptive KV benchmark summary

schema_version: 1

git_revision: 7666c6173e837801d66bc46e7a29d3c0ab147283


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 303.72 | 0.003293 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 308.52 | 0.003241 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 314.98 | 0.003175 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 314.61 | 0.003179 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 219.99 | 0.004546 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 220.47 | 0.004536 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 221.63 | 0.004512 | 64 | True | 71335936 | 6 | 1 |
