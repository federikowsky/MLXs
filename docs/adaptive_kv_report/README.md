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

This package was written from repository source material, including the retained Adaptive KV design document, freeze summaries, real-workload evaluation, synthetic benchmark artifacts, parity investigations, and the retained benchmark harness. It reflects the retained runtime-families baseline: a generic semantic core, an explicit runtime-family layer, and **three** retained exact concrete substrates—Family A `full_kv`, Family B `windowed_kv`, and Family C `hybrid_state`—with model adapters binding standard `ministral3` (B) and standard text-only `qwen3_5` (C) where the capability gate allows. The principal synthetic and real-workload tables remain Llama-centric; multimodal paths, `compile_decode`, legacy `quantized_kv_start`, and external cache reuse stay unsupported. The older development documents remain useful as source records, but this package is the cleaner standalone reader-facing form.
