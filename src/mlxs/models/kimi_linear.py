"""Kimi Linear: hybrid MLA + gated-delta (linear) attention with optional MoE.

Ported from mlx_lm. Implements ModelProtocol. Alternating MLA and delta
(linear) layers per linear_attn_config["kda_layers"]; MoE via SwitchGLU.
Imports: mlxs.cache, mlxs.layers, mlxs.models.base.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.gated_delta import gated_delta_update
from mlxs.layers.mla import MultiLinear
from mlxs.layers.moe import SwitchGLU
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Kimi Linear config (from config.json / mlx_lm)."""

    model_type: str = "kimi_linear"
    vocab_size: int = 128256
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    intermediate_size: int = 11008
    head_dim: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    linear_attn_config: dict[str, Any] | None = None
    model_max_length: int = 32768
    num_experts: int = 0
    moe_intermediate_size: int = 1408
    kv_lora_rank: int = 512
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    mla_use_nope: bool = False
    num_experts_per_token: int = 1
    num_shared_experts: int = 0
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        sig = inspect.signature(cls)
        return cls(**{k: v for k, v in params.items() if k in sig.parameters})


class _KimiMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(
        self,
        args: ModelArgs,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        dim = hidden_size if hidden_size is not None else args.hidden_size
        hidden = intermediate_size if intermediate_size is not None else args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def _group_expert_select(
    gates: mx.array,
    bias: mx.array | None,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> tuple[mx.array, mx.array]:
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates)
    elif score_function == "softmax":
        scores = mx.softmax(gates, axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE router activation '{score_function}'")

    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, scores * routed_scaling_factor


class _KimiSparseMoE(nn.Module):
    """Sparse MoE with SwitchGLU and optional shared expert."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        if experts is None or experts <= 0:
            raise ValueError("num_experts must be > 0 for MoE layers")
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, args.moe_intermediate_size, experts)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if args.num_shared_experts and args.num_shared_experts > 0:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = _KimiMLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        scores = self.gate(x)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )
        out = self.switch_mlp(x, inds)
        out = (out * weights[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class _KimiMLAAttention(nn.Module):
    """MLA (compressed KV + RoPE) attention."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or args.head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            args.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(args.kv_lora_rank, self.v_head_dim, self.num_heads)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class _ShortConv1d(nn.Module):
    """Depthwise 1D conv + SiLU for delta attention inputs."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: mx.array | None,
        mask: mx.array | None,
        lengths: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = conv_input[:, -n_keep:, :]

        return out, new_state


class _KimiDeltaAttention(nn.Module):
    """Gated-delta (linear) attention with short conv and SSM state."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        cfg = args.linear_attn_config or {}
        self.layer_idx = layer_idx
        self.num_heads = cfg.get("num_heads", args.num_attention_heads)
        self.head_dim = cfg.get("head_dim", args.head_dim)
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size

        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = _ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = _ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = _ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.expand_dims(
            mx.log(mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))),
            (0, 1, 3),
        )
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state = cache[0]
            k_state = cache[1]
            v_state = cache[2]
            ssm_state = cache[3]
            lengths = cache.lengths
        else:
            q_state = None
            k_state = None
            v_state = None
            ssm_state = None
            lengths = None

        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state = s
            k_state = s
            v_state = s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        inv_scale = self.scale
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(B, T, self.num_heads, self.head_dim)
        b_logits = self.b_proj(x).reshape(B, T, self.num_heads)

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask,
            use_kernel=False,
        )

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        gate = self.g_b_proj(self.g_a_proj(x)).reshape(B, T, self.num_heads, self.head_dim)
        out = (
            self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim)) * mx.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)


class _KimiDecoderLayer(nn.Module):
    """Single decoder block: MLA or delta attention + MLP or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        kda_layers = (args.linear_attn_config or {}).get("kda_layers", [])
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = _KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = _KimiMLAAttention(args)

        if (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = _KimiSparseMoE(args)
        else:
            self.mlp = _KimiMLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        y = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + y
        z = self.mlp(self.post_attention_layernorm(h))
        return h + z


class _KimiLinearModel(nn.Module):
    """Backbone: embed + alternating MLA/delta layers + norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [_KimiDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        kda_layers = (args.linear_attn_config or {}).get("kda_layers", [])
        self.ssm_idx = (kda_layers[0] - 1) if kda_layers else 0
        self.attn_idx = 0
        for i in range(len(self.layers)):
            if (i + 1) not in kda_layers:
                self.attn_idx = i
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache | None] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, cache[self.attn_idx], return_array=True)

        for layer, layer_cache in zip(self.layers, cache, strict=True):
            mask = ssm_mask if layer.is_linear else attn_mask
            h = layer(h, mask=mask, cache=layer_cache)

        return self.norm(h)


class Model(nn.Module):
    """Kimi Linear: ModelProtocol implementation."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = _KimiLinearModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[Any] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        caches: list[KVCache | ArraysCache] = []
        for layer in self.model.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        weights = {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, _KimiSparseMoE):
                src_prefix = f"{prefix}.block_sparse_moe"
                dst_prefix = f"{prefix}.mlp"
                for src, dst in [
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ]:
                    key = f"{src_prefix}.experts.0.{src}.weight"
                    if key in weights:
                        stacked = [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.weight")
                            for i in range(self.args.num_experts)
                        ]
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(stacked)

                for name in ("gate_proj", "up_proj", "down_proj"):
                    src_key = f"{src_prefix}.shared_experts.{name}.weight"
                    if src_key in weights:
                        weights[f"{dst_prefix}.shared_experts.{name}.weight"] = weights.pop(
                            src_key
                        )

                gate_key = f"{src_prefix}.gate.weight"
                if gate_key in weights:
                    weights[f"{dst_prefix}.gate.weight"] = weights.pop(gate_key)

                bias_key = f"{src_prefix}.gate.e_score_correction_bias"
                if bias_key in weights:
                    weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(bias_key)

            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, _KimiDeltaAttention):
                attn_prefix = f"{prefix}.self_attn"
                for src_name, dst_name in (
                    ("q_conv1d", "q_conv"),
                    ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ):
                    src_key = f"{attn_prefix}.{src_name}.weight"
                    if src_key in weights:
                        w = weights.pop(src_key)
                        if w.ndim == 3:
                            w = w.moveaxis(2, 1)
                        weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
                dt_key = f"{attn_prefix}.dt_bias"
                if dt_key in weights and weights[dt_key].ndim > 1:
                    weights[dt_key] = mx.reshape(weights[dt_key], (-1,))

            attn_prefix = f"{prefix}.self_attn"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = self.args.qk_nope_head_dim or self.args.head_dim
                v_head = self.args.v_head_dim or self.args.head_dim
                head_dim = qk_nope + v_head
                num_heads = self.args.num_attention_heads

                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, qk_nope:, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights

    def parameters(self) -> dict[str, Any]:
        return dict(self.items())
