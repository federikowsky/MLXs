# Adaptive KV benchmark summary

schema_version: 1

git_revision: fd20156f037dba4151d93875d72600c0c931cf6c


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| long_static | non_adaptive | 128.22 | 0.007799 | 256 | None |  | 0 | 0 |
| long_static | adaptive_full | 125.15 | 0.007991 | 256 | True | 146685952 | 0 | 0 |
| long_static | adaptive_full | 126.69 | 0.007893 | 256 | True | 146685952 | 0 | 0 |
| long_static | adaptive_soft | 122.23 | 0.008182 | 256 | True | 89800704 | 0 | 0 |
| long_static | adaptive_soft | 121.61 | 0.008223 | 256 | True | 89800704 | 0 | 0 |
| hard_pressure_context | non_adaptive | 127.70 | 0.007831 | 256 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 123.37 | 0.008106 | 256 | True | 146685952 | 0 | 0 |
| hard_pressure_context | adaptive_full | 124.29 | 0.008046 | 256 | True | 146685952 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 118.34 | 0.008451 | 256 | True | 89800704 | 0 | 0 |
| hard_pressure_context | adaptive_soft | 118.48 | 0.008440 | 256 | True | 89800704 | 0 | 0 |
