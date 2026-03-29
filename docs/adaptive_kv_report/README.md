# Adaptive KV Report Package

This directory contains a standalone technical report package for Adaptive KV in MLXs.

Archival note:

This package is **historical source material** for Adaptive KV **V1** / `FULL`-`COMPRESSED`-`EVICTED` evaluation-era benchmarks and narrative. It is **not** the canonical description of the later **exact adaptive execution-pack** frozen baseline, and it is **not** the record of the current **mainline pivot** back toward pragmatic tiers.

Reading order:

1. [main.md](./main.md)  
   Primary document for the V1 report era. Self-contained for that era’s problem, design, guarantees, evaluation tables, and conclusions.
2. [appendix_experimental_setup.md](./appendix_experimental_setup.md)  
   Detailed benchmark and workload setup, retained scope, and interpretation conventions.
3. [appendix_results.md](./appendix_results.md)  
   Expanded tables for synthetic benchmarks, real workloads, and later regression checks.
4. [appendix_correctness.md](./appendix_correctness.md)  
   Correctness envelope, parity evidence, and the distinction between exact semantic guarantees and compression-fidelity limitations.

For the **frozen** exact adaptive execution-pack baseline, technical freeze, throughput limits, and rejected closure attempts, use:

1. [../adaptive_kv_execution_pack_track_archival.md](../adaptive_kv_execution_pack_track_archival.md) — track archival decision and next direction
2. [../adaptive_kv_turboquant_first.spec.md](../adaptive_kv_turboquant_first.spec.md) — canonical technical specification (filename retained)
3. [../dev/repo_mapping_kv_cache.md](../dev/repo_mapping_kv_cache.md) — repo integration map

This V1 package remains useful as historical reader-facing background; it must not be read as the final word on current product strategy or on the execution-pack era.
