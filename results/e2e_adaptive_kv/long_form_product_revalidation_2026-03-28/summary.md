# Adaptive KV Long-Form Product E2E Revalidation Summary

Artifacts:
- `evaluation.json`: structured scenario, regime, transcript, and metric data
- `transcripts.md`: full baseline and Adaptive KV long-form conversation transcripts
- `run_eval.py`: reproducible runner used for this long-form revalidation

Main model in the matrix:
- `unsloth/Llama-3.2-1B-Instruct` via the fully cached local snapshot — supported Family A `full_kv`

Model selection note:
- A stronger Family A candidate, `meta-llama/Llama-3.2-3B-Instruct`, was not usable in this environment because the hub path was gated and the local cache was incomplete.
- Family C was omitted from the main matrix because the supported `qwen3_5` checkpoints available here still did not behave like credible product-chat baselines.
- Family B was omitted because no practical chat-capable Ministral-family checkpoint was staged locally for a serious long-form comparison.

Regime design:
- Comfortable: average baseline projected peak KV footprint `0.58x` the Adaptive hard budget
- Borderline: average baseline projected peak KV footprint `1.94x` the Adaptive hard budget
- Stressed: average baseline projected peak KV footprint `4.71x` the Adaptive hard budget

High-level result:
- The execution-pack baseline does not materially change the prior product story.
- Adaptive KV remains primarily an operational serving architecture rather than a guaranteed conversational-quality improver.
- In this longer-form run, Adaptive KV was slower in every regime and did not show a clear user-visible quality advantage over baseline.
- Under borderline and stressed envelopes, Adaptive KV did show real pressure-managed behavior: soft/hard transitions, evictions, recomputes, and replay-backed continuation.

Comfortable regime:
- Aggregate: baseline `25.70 tok/s`, adaptive `21.80 tok/s`
- Operationally: no evictions, no recomputes; Adaptive mostly stayed `normal`, with late `soft` on the longer scenarios
- User-visible read:
  - Scenario A: baseline slightly better; Adaptive was more repetitive and less clean in the final synthesis
  - Scenario B: roughly equivalent; both modes were repetitive and weak on the math, and final answers matched exactly
  - Scenario C: roughly equivalent to slight baseline edge; Adaptive was not more helpful
  - Scenario D: baseline slightly better; Adaptive drifted more in the final dinner plan

Borderline regime:
- Aggregate: baseline `24.45 tok/s`, adaptive `17.36 tok/s`
- Operationally: Adaptive entered visible pressure, with `174` total evictions and `21` recomputes across the four scenarios
- User-visible read:
  - Scenario A: roughly equivalent; Adaptive preserved a usable structure under pressure but did not improve the plan
  - Scenario B: equivalent; all turns matched exactly despite hard pressure
  - Scenario C: baseline slightly better; Adaptive felt less anchored to the revised budget and reuse constraints
  - Scenario D: baseline better; Adaptive lost some cleanup/timing discipline as pressure increased

Stressed regime:
- Aggregate: baseline `22.96 tok/s`, adaptive `14.29 tok/s`
- Operationally: all four scenarios entered hard pressure; Adaptive recorded `580` evictions and `24` recomputes
- User-visible read:
  - Scenario A: baseline slightly better; Adaptive stayed usable but more generic
  - Scenario B: equivalent; all turns matched exactly even under heavy pressure
  - Scenario C: baseline better; Adaptive was slower and not more useful
  - Scenario D: baseline better; Adaptive preserved the broad dinner-planning thread but gave a weaker final synthesis

What a real user would notice:
- Comfortable: mostly slower responses, not better answers
- Borderline: still mostly slower responses, with only occasional parity under pressure
- Stressed: the baseline answers were usually as good or better on this checkpoint, but Adaptive KV continued to serve through hard pressure with explicit evictions/recompute instead of pretending pressure did not exist

Operational caveats:
- `resident_bytes` still overshot the nominal hard budget in the longer borderline and stressed runs, so this remains a pressure-managed serving path rather than a strict hard-cap proof from end-of-turn snapshots
- The model itself is still small and imperfect; many answers in both modes were mediocre, which limits how much user-visible upside Adaptive KV can demonstrate on this checkpoint alone

Bottom line:
- The retained execution-pack baseline strengthens the serving substrate and keeps the system operationally controlled under heavier context pressure.
- It does not, on this long-form product run, demonstrate a consistent user-visible conversational quality win over normal baseline inference.
- The honest product framing remains: Adaptive KV is primarily valuable as an exact, pressure-managed serving architecture, with user-visible quality benefits being checkpoint- and regime-dependent rather than guaranteed.
