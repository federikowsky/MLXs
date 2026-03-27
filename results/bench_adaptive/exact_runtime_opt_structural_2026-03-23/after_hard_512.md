# Adaptive KV benchmark summary

schema_version: 1

git_revision: c74f2e725d7c7895371065b129fe119d86566764


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 212.73 | 0.004701 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 220.19 | 0.004541 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 218.02 | 0.004587 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_full | 224.71 | 0.004450 | 64 | True | 65945600 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 208.37 | 0.004799 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 201.44 | 0.004964 | 64 | True | 41058304 | 1 | 1 |
| hard_pressure_context | adaptive_hard | 204.16 | 0.004898 | 64 | True | 41058304 | 1 | 1 |
