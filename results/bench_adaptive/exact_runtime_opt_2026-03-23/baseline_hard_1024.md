# Adaptive KV benchmark summary

schema_version: 1

git_revision: 7666c6173e837801d66bc46e7a29d3c0ab147283


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 308.81 | 0.003238 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 289.60 | 0.003453 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 296.84 | 0.003369 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_full | 307.88 | 0.003248 | 64 | True | 124665856 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 212.61 | 0.004703 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 204.63 | 0.004887 | 64 | True | 71335936 | 6 | 1 |
| hard_pressure_context | adaptive_hard | 214.79 | 0.004656 | 64 | True | 71335936 | 6 | 1 |
