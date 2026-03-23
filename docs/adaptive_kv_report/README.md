# Adaptive KV Report Package

This directory contains a standalone technical report package for Adaptive KV in MLXs.

Reading order:

1. [main.md](./main.md)  
   Primary document. It is intended to be self-contained and should be sufficient for understanding the problem, design, guarantees, evaluation, and final conclusions.
2. [appendix_experimental_setup.md](./appendix_experimental_setup.md)  
   Detailed benchmark and workload setup, retained scope, and interpretation conventions.
3. [appendix_results.md](./appendix_results.md)  
   Expanded tables for synthetic benchmarks, real workloads, and later regression checks.
4. [appendix_correctness.md](./appendix_correctness.md)  
   Correctness envelope, parity evidence, and the distinction between exact semantic guarantees and compression-fidelity limitations.

This package was written from repository source material, including the retained Adaptive KV design document, freeze summaries, real-workload evaluation, synthetic benchmark artifacts, parity investigations, and the retained benchmark harness. The older development documents remain useful as source records, but this package is the cleaner standalone reader-facing form.
