# Adaptive KV Exact Split-Pack Attention Compute Path Summary

This run tested one low-level exact compute-path idea on top of the retained execution-pack baseline:

- cache a stable dense prefix for the few-pack unsampled path
- feed `mx.fast.scaled_dot_product_attention` with `dense_prefix + live tail`
- preserve the retained architecture, cadence, and family semantics

Result:

- the experiment was exact but not retainable
- calm-long A/B/C all regressed versus the retained baseline
- pressured sanity also regressed materially
- the slowdown stayed concentrated in `attention.fast_path_ns`, especially `attention.fast_path_sdpa_ns`

Comfortable long vs retained baseline:

- `llama_long_soft_96_normal`: `147.92 tok/s` vs `233.60 tok/s` (`-36.68%`)
- `ministral3_long_soft_96_normal`: `193.91 tok/s` vs `258.50 tok/s` (`-24.98%`)
- `qwen35_long_soft_96_normal`: `134.07 tok/s` vs `185.31 tok/s` (`-27.65%`)

Pressured sanity vs retained baseline:

- `llama_long_hard`: `48.40 tok/s` vs `71.36 tok/s` (`-32.17%`)
- `ministral3_long_soft_96`: `92.41 tok/s` vs `114.57 tok/s` (`-19.34%`)
- `qwen35_long_soft_96`: `79.50 tok/s` vs `100.31 tok/s` (`-20.74%`)
- `qwen35_short_hard_hybrid`: `98.37 tok/s` vs `138.39 tok/s` (`-28.92%`)

Attribution highlights:

- baseline calm-long `attention.fast_path_concat_ns` was already tiny
- the experiment did not remove the real bottleneck
- instead it made `attention.fast_path_sdpa_ns` worse:
  - Family A: `661.3k ns -> 814.7k ns`
  - Family B: `721.5k ns -> 867.0k ns`
  - Family C: `1.476M ns -> 1.644M ns`

Conclusion:

- the split-pack residual is not being driven by Python-side pack concatenation alone
- a cached dense prefix is not a valid retained compute-path closure
- this direction was rejected; the exact adaptive execution-pack track was subsequently **archived** as a mainline strategy (see `docs/adaptive_kv_execution_pack_track_archival.md`), with follow-on work pivoting to pragmatic `FULL` / `COMPRESSED` / `EVICTED`
