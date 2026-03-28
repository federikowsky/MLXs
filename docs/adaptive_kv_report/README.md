# Adaptive KV Report Package

This directory contains a standalone technical report package for Adaptive KV in MLXs.

Archival note:

This package is retained as **historical source material** for the earlier Adaptive KV V1 / `FULL`-`COMPRESSED`-`EVICTED` report baseline and related investigations. It is **not** the canonical architecture source of truth for the completed TurboQuant-first branch.

Reading order:

1. [main.md](./main.md)  
   Primary document. It is intended to be self-contained and should be sufficient for understanding the problem, design, guarantees, evaluation, and final conclusions.
2. [appendix_experimental_setup.md](./appendix_experimental_setup.md)  
   Detailed benchmark and workload setup, retained scope, and interpretation conventions.
3. [appendix_results.md](./appendix_results.md)  
   Expanded tables for synthetic benchmarks, real workloads, and later regression checks.
4. [appendix_correctness.md](./appendix_correctness.md)  
   Correctness envelope, parity evidence, and the distinction between exact semantic guarantees and compression-fidelity limitations.

For the final retained TurboQuant-first branch architecture, validation stance, and archival position, use:

1. [../adaptive_kv_turboquant_first.spec.md](../adaptive_kv_turboquant_first.spec.md)
2. [../dev/repo_mapping_kv_cache.md](../dev/repo_mapping_kv_cache.md)

This package remains useful as historical reader-facing background, but it should not be read as the final branch specification.
