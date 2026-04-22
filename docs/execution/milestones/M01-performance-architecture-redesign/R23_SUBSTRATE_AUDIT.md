# M01.R23 Forward-Step Substrate Audit

Updated: 2026-04-20

## Objective

Audit the decisive single-request weakest-case boundary below helper choreography and determine whether a bounded substrate patch is justified.

Decisive target:

- model: `mlx-community/Llama-3.2-1B-Instruct-4bit`
- prompt: `256`
- path: prompt-tail handoff into first decode step after prefill

This audit exists because `R22` already proved that a near-`mlx_lm` resident helper token loop is not enough.

## Governing accepted evidence

- Accepted weakest-case ledger stays:
  - `Llama-3.2-1B 256 decode 0.99997x`
- `R20` remote decomposition showed helper-path `schedule_next` / forward-build was dominant on the accepted helper surface:
  - MLXs `schedule_next_per_token_s ~0.00302744`
  - `mlx_lm schedule_next_per_token_s ~0.00034372`
- `R22` rejected the next adjacent helper-only redesign:
  - remote absolute MLXs decode throughput stayed slightly regressive on both `256` and `2048`

## Current substrate map

### MLXs

- model: `mlxs.models.llama.Model`
- cache: `mlxs.cache.kv.KVCache`
- attention mask: `mlxs.cache.attention_mask.create_attention_mask()`
- attention dispatch: `mlxs.layers.attention.scaled_dot_product_attention()`
- rope init: `mlxs.layers.rope.initialize_rope()`
- Layer 1 decode owner:
  - `src/mlxs/runtime_core/prefill.py`
  - `src/mlxs/runtime_core/decode.py`
  - `src/mlxs/runtime_core/greedy.py`

### `mlx_lm`

- model: `mlx_lm.models.llama.Model`
- cache: `mlx_lm.models.cache.KVCache`
- attention mask: `mlx_lm.models.base.create_attention_mask()`
- attention dispatch: `mlx_lm.models.base.scaled_dot_product_attention()`
- rope init: `mlx_lm.models.rope_utils.initialize_rope()`
- comparable generation owner:
  - `mlx_lm.generate.generate_step()`

### Structural note

The steady-state Llama model/cache codepaths are nearly identical by inspection. The next likely tax is therefore not broad model-wrapper divergence, but the boundary between:

1. prompt prefill completion,
2. first cached decode forward,
3. first cache growth/use on the decode side,
4. first-step stream/eval discipline.

## Probe method

All first-pass probes must stay local and non-promoting.

### Required local probes

1. Interleaved first-decode boundary probe on `Llama-3.2-1B 256`
   - alternate MLXs and `mlx_lm` in the same process order as much as practical
   - measure:
     - prompt-tail handoff into first decode
     - first cached model forward
     - first `KVCache.update_and_fetch()`
     - attention-mask creation
     - stream/eval boundary cost

2. Interleaved MLXs-vs-MLXs control/candidate probe
   - only after one bounded patch exists
   - absolute MLXs throughput is the primary signal

3. Full local Class A rerun on `Llama-3.2-1B 256`
   - only if the component probe clears the evidence gate

4. Guardrails after positive decisive local signal
   - `Llama-3.2-1B 2048`
   - `Qwen2.5-1.5B 256`
   - `Llama-3.2-3B 256`

### Early no-patch reads already collected

- steady-state cached forward medians are near parity on a simple local read:
  - MLXs `~0.01273s`
  - `mlx_lm ~0.01267s`
- one simple first-decode-after-prefill read was directionally worse for MLXs:
  - MLXs `~0.01480s`
  - `mlx_lm ~0.00989s`
- isolated synthetic `KVCache.update_and_fetch()` first-call asymmetry was not stable under order reversal, so it is not yet actionable

Interpretation:

- broad steady-state model forward is not the current primary suspect
- the prompt-tail to first-decode boundary is the first substrate target
- isolated first-call cache warmup effects are not enough evidence by themselves

## Evidence gate

Patch allowed only if one substrate component is both isolated and strong enough.

Default gate:

- one component on the decisive `Llama-3.2-1B 256` boundary must show:
  - median MLXs-vs-`mlx_lm` delta `>= 10%`
  - and absolute median delta `>= 0.5 ms`
  - and the same direction across at least `3` interleaved probe batches
- ownership must be clear enough to keep the first patch bounded to at most:
  - Layer 1 handoff
  - cache update/fetch behavior
  - or first-step stream/eval discipline

Local-to-remote promotion gate after a patch exists:

- local MLXs control/candidate on full Class A `Llama-3.2-1B 256` must improve absolute decode throughput by at least `3%`
- local exact-token and exact-logit parity must hold on the decisive path
- only then run remote control/candidate validation

Remote promotion rule:

- promote only if remote absolute MLXs decode throughput improves vs remote control on the decisive case
- ratio-only wins caused by `mlx_lm` drift do not count

## Immediate next slice

Run an interleaved local probe focused on the prompt-tail to first-decode boundary and decide whether the first bounded redesign slice belongs to:

1. prefill-to-first-decode handoff in Layer 1
2. first-step cache update/fetch allocation/growth
3. first-step stream/eval ownership

If the gate fails, close `R23` negatively and pivot immediately to the batch-first fallback front.

## Result

Artifacts:

- local: `/tmp/mlxs_m01_r23_forward_step_probe_local_20260420_1.json`
- remote: `/tmp/mlxs_m01_r23_forward_step_probe_remote_20260420_1.json`

Key local summary:

- current MLXs vs comparable `mlx_lm` total prompt-tail to next-ready boundary:
  - ratio `~0.81999x` (`MLXs` faster)
- current MLXs vs comparable `mlx_lm` isolated first-decode-forward component:
  - ratio `~2.67749x` (`MLXs` slower)
- no-patch integrated boundary probe vs current MLXs total boundary:
  - ratio `~1.77837x` (`probe` worse)

Key remote summary:

- current MLXs vs comparable `mlx_lm` total prompt-tail to next-ready boundary:
  - ratio `~0.87849x` (`MLXs` faster)
- current MLXs vs comparable `mlx_lm` isolated first-decode-forward component:
  - ratio `~1.78472x` (`MLXs` slower)
- no-patch integrated boundary probe vs current MLXs total boundary:
  - ratio `~1.92319x` (`probe` worse)

Structural note:

- current MLXs cache capacity after the tail prompt step stayed `512`
- comparable `mlx_lm` cache capacity stayed `256` before the first decode step and grew to `512` on that step
- MLXs therefore already removes the first-decode cache-growth hit on this boundary

Conclusion:

- the isolated first-decode-forward component is slower in MLXs
- but the current full prompt-tail / first-decode boundary is already better than the comparable `mlx_lm` boundary in total wall time
- the first bounded no-patch integrated redesign makes the total boundary materially worse
- `R23` therefore does not expose a clear bounded single-request substrate patch worth implementing

Classification:

- negative audit
- pivot to the batch-first fallback front
