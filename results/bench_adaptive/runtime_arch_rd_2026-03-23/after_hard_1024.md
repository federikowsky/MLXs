# Adaptive KV benchmark summary

schema_version: 1

git_revision: 83b88dbff34e5b07099eabf0db1dae8334920ad6


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 296.22 | 0.003376 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 307.13 | 0.003256 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 305.40 | 0.003274 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 310.62 | 0.003219 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 221.61 | 0.004512 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 217.95 | 0.004588 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 219.56 | 0.004555 | 64 | True | 71335936 | 6 | 1 |
