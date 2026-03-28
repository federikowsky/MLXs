# Adaptive KV Product-Grade E2E Conversational Evaluation Summary

Artifacts:
- `evaluation.json`: structured scenario, regime, transcript, and metric data
- `transcripts.md`: full baseline and Adaptive KV conversation transcripts
- `run_eval.py`: reproducible runner used for this product-grade evaluation

Main model in the matrix:
- `unsloth/Llama-3.2-1B-Instruct` — supported Family A `full_kv`

Family selection note:
- Family C was not included in the main product-grade matrix because the supported `qwen3_5` checkpoints available in this environment were not usable chat baselines for a realistic product comparison.
- Probed but rejected for the main matrix:
  - `Qwen/Qwen3.5-0.8B`
  - `Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled`
  - `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated`

Regime design:
- Comfortable: average baseline projected peak KV footprint `0.43x` the configured Adaptive hard budget
- Borderline: average baseline projected peak KV footprint `1.07x` the configured Adaptive hard budget
- Stressed: average baseline projected peak KV footprint `3.19x` the configured Adaptive hard budget

High-level result:
- Adaptive KV did not produce a clear user-visible conversational quality win in this product-grade run.
- Its main effect was operational: pressure tracking, evictions, recomputes, and a slower bounded-memory serving path under tighter envelopes.
- On the tested Llama 3.2 instruct checkpoint, baseline and Adaptive KV diverged on the first turn of most scenarios, so this was not an “identical answers, different runtime” story.

Comfortable regime:
- No pressure, no evictions, no recomputes.
- Adaptive KV was still slower across all four scenarios.
- User-visible quality was mixed but not better overall than baseline.

Borderline regime:
- Two scenarios stayed in soft pressure only; two escalated into hard pressure.
- Adaptive KV introduced `16` total evictions and `6` recomputes across the four scenarios.
- One scenario (`C_iterative_assistant_workflow`) matched baseline exactly; the others diverged.
- No clear user-visible quality gain appeared, but operational pressure handling became visible.

Stressed regime:
- All four scenarios entered hard pressure.
- Adaptive KV introduced `275` total evictions and `16` recomputes across the four scenarios.
- Two scenarios still diverged from baseline in their final answers; one matched exactly end-to-end; one matched only at the final answer.
- User-visible quality did not clearly improve under stress. In one scenario it became more repetitive than baseline.

Latency / throughput:
- Comfortable aggregate: baseline `29.48 tok/s`, adaptive `23.68 tok/s`
- Borderline aggregate: baseline `28.77 tok/s`, adaptive `20.94 tok/s`
- Stressed aggregate: baseline `26.94 tok/s`, adaptive `14.34 tok/s`

Operational caveat:
- In this run, `resident_bytes` did not stay below the configured hard budget in borderline or stressed regimes.
- The retained system clearly applied pressure, evictions, and replay/recompute, but the observed resident-byte snapshots overshot the nominal hard-budget settings.
- That means the practical value shown here is pressure-managed operation, not a strict hard-cap proof.

Product interpretation:
- On this hardware, with this stronger chat-capable Family A model, Adaptive KV is best understood here as an operational serving mechanism, not a conversational quality improvement.
- If the baseline fits comfortably, a real user would mainly notice slower responses.
- Under tighter serving envelopes, Adaptive KV gives more explicit pressure management, but this run did not show a better practical answer quality outcome for the user.
