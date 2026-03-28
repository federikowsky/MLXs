# Adaptive KV Real E2E Conversational Evaluation Summary

Artifacts:
- `evaluation.json`: structured per-turn metrics and outputs
- `transcripts.md`: full baseline vs Adaptive KV conversation transcripts

Method:
- Two real supported models were evaluated on the same four multi-turn conversational scenarios in baseline and Adaptive KV modes.
- The canonical conversation history was built from baseline assistant outputs so both modes saw the same prompt on every turn.
- Settings were matched: deterministic decode (`temperature=0.0`, `seed=1234`), `max_tokens=128`, same system prompt, same turn order.

Models:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — Family A
- `Qwen/Qwen3.5-0.8B` — Family C

High-level findings:
- Adaptive KV preserved practical conversational behavior relative to baseline more than it improved it.
- On the chat-capable Family A model, Adaptive KV responses were usually identical or near-identical to baseline but slower in every scenario.
- On the chosen Family C text-only checkpoint, both baseline and Adaptive KV produced largely unusable multilingual garbage; Adaptive KV preserved that behavior but did not improve it.
- The main user-visible difference in this evaluation was latency, not answer quality.

TinyLlama / Family A:
- Scenario A: 6 turns, 4 identical, baseline 22.149s, adaptive 35.873s, adaptive evictions 15, recomputes 4, max resident 24,352,768 bytes
- Scenario B: 5 turns, 4 identical, baseline 18.136s, adaptive 27.237s, adaptive evictions 8, recomputes 3, max resident 20,748,288 bytes
- Scenario C: 5 turns, 4 identical, baseline 17.648s, adaptive 25.221s, adaptive evictions 7, recomputes 3, max resident 19,757,056 bytes
- Scenario D: 8 turns, 6 identical, baseline 29.595s, adaptive 52.794s, adaptive evictions 33, recomputes 6, max resident 30,164,992 bytes
- Overall: 18/24 turns exactly matched; baseline 35.10 tok/s aggregate vs adaptive 21.77 tok/s

Qwen3.5 / Family C:
- Scenario A: 6 turns, 5 identical, baseline 39.561s, adaptive 50.899s, adaptive evictions 37, recomputes 6, max resident 45,096,960 bytes
- Scenario B: 5 turns, 4 identical, baseline 32.516s, adaptive 41.175s, adaptive evictions 27, recomputes 5, max resident 42,049,536 bytes
- Scenario C: 5 turns, 5 identical, baseline 33.616s, adaptive 44.965s, adaptive evictions 26, recomputes 5, max resident 41,410,560 bytes
- Scenario D: 8 turns, 8 identical, baseline 57.438s, adaptive 81.130s, adaptive evictions 68, recomputes 8, max resident 52,764,672 bytes
- Overall: 22/24 turns exactly matched; baseline 18.83 tok/s aggregate vs adaptive 14.08 tok/s

Practical interpretation:
- For these realistic multi-turn chats at the tested lengths, Adaptive KV did not produce better user-visible answers than the normal baseline.
- Its value in this run was operational: it maintained exact outputs while enforcing explicit memory budgets and absorbing pressure through evictions/recompute.
- If the baseline fits comfortably in memory, a real user would mostly notice slower responses, not better answers.
- If the deployment target is memory-constrained or conversation lengths exceed the normal baseline’s comfortable KV footprint, Adaptive KV remains the more operationally robust path.

Caveats:
- The Family C checkpoint used here is supported by the retained branch but did not behave like a usable instruction-tuned chat model, so its user-quality comparison is mostly a preservation-of-behavior result, not a product-quality endorsement.
- All turns in both modes consumed the full `max_tokens=128` cap, which reflects small-model chat quality limits in this setup rather than an Adaptive-KV-specific stopping issue.
