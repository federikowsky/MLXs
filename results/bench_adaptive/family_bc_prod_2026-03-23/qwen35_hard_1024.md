# Adaptive KV benchmark summary

schema_version: 1

git_revision: cb3f751ebc45e834acf0b0739e1a3ce0bc87df75


| scenario | baseline | tok/s | s/tok | gen_tok | ref_ok | res_B | evict | recomp_rq |
|----------|----------|-------|-------|---------|--------|-------|-------|-----------|
| hard_pressure_context | non_adaptive | 94.93 | 0.010534 | 64 | None |  | 0 | 0 |
| hard_pressure_context | adaptive_full | 96.99 | 0.010311 | 64 | True | 61964288 | 0 | 0 |
| hard_pressure_context | adaptive_hard | 70.51 | 0.014183 | 64 | True | 46727168 | 1 | 1 |
