"""HunYuan MoE model — ModelProtocol.

Dense + MoE (optional shared expert), CLA (cross-layer attention sharing),
QK-norm, Dynamic NTK Alpha RoPE. Port from mlx_lm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.models.base import BaseModelArgs


def _int_or_list(arg: int | list[int], idx: int) -> int:
    if isinstance(arg, list):
        return arg[idx]
    return arg


# ----- Model args -----


@dataclass
class ModelArgs(BaseModelArgs):
    """HunYuan model configuration (flat, from config.json)."""

    model_type: str = "hunyuan"
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    attention_bias: bool = False
    moe_topk: int = 2
    num_experts: int = 8
    num_shared_expert: int | list[int] = 1
    use_mixed_mlp_moe: bool = True
    use_qk_norm: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_cla: bool = True
    cla_share_factor: int = 2
    moe_intermediate_size: int | list[int] | None = None
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.rope_scaling is not None:
            required = {"factor", "type"}
            if not required.issubset(self.rope_scaling):
                raise ValueError(f"rope_scaling must contain {required}")


# ----- RoPE (model-specific: NTK alpha scaling) -----


class DynamicNTKAlphaRoPE(nn.Module):
    """RoPE with NTK alpha scaling (HunYuan). Fixed freqs from base * alpha^(dims/(dims-2))."""

    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        scaling_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.dims = dims
        base_scaled = base * (scaling_alpha ** (dims / (dims - 2)))
        self._freqs = base_scaled ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


# ----- Attention -----


class Attention(nn.Module):
    """Multi-head attention with optional QK-norm and CLA (shared KV from previous layer)."""

    def __init__(self, kv_proj: bool, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = dim // self.n_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=args.attention_bias)
        if kv_proj:
            self.k_proj = nn.Linear(
                dim, self.n_kv_heads * head_dim, bias=args.attention_bias
            )
            self.v_proj = nn.Linear(
                dim, self.n_kv_heads * head_dim, bias=args.attention_bias
            )
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=args.attention_bias)
        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)
        scaling = args.rope_scaling or {}
        self.rope = DynamicNTKAlphaRoPE(
            head_dim,
            base=args.rope_theta,
            scaling_alpha=float(scaling.get("alpha", 1.0)),
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
        kv_states: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        if kv_states is None:
            keys = self.k_proj(x)
            values = self.v_proj(x)
            kv_states = (keys, values)
        else:
            keys, values = kv_states

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), kv_states


# ----- MLP -----


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Gate(nn.Module):
    """Router for MoE."""

    def __init__(self, dim: int, num_experts: int) -> None:
        super().__init__()
        self.wg = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.wg(x)


# ----- MoE block -----


class MoeBlock(nn.Module):
    """Top-k MoE with SwitchGLU experts and optional shared MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0) -> None:
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.intermediate_size
        self.use_shared_mlp = args.use_mixed_mlp_moe

        if args.use_mixed_mlp_moe:
            num_shared = _int_or_list(args.num_shared_expert, layer_idx)
            self.shared_mlp = MLP(dim, int(intermediate_size * num_shared))

        self.num_experts = args.num_experts
        self.top_k = _int_or_list(args.moe_topk, layer_idx)
        self.gate = Gate(dim, self.num_experts)

        expert_hidden = intermediate_size
        if args.moe_intermediate_size is not None:
            expert_hidden = _int_or_list(args.moe_intermediate_size, layer_idx)

        self.switch_mlp = SwitchGLU(dim, expert_hidden, self.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)

        if self.use_shared_mlp:
            y = y + self.shared_mlp(x)
        return y


# ----- Decoder layer -----


class DecoderLayer(nn.Module):
    """Pre-norm block: attention (with optional CLA) + MLP or MoE."""

    def __init__(self, args: ModelArgs, kv_proj: bool, layer_idx: int = 0) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(kv_proj, args)
        if args.num_experts == 1:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = MoeBlock(args, layer_idx)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
        shared_kv_states: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        r, shared_kv_states = self.self_attn(
            self.input_layernorm(x), mask, cache, shared_kv_states
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, shared_kv_states


# ----- Backbone -----


class HunYuanModel(nn.Module):
    """HunYuan transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args=args,
                kv_proj=(not args.use_cla) or (i % args.cla_share_factor == 0),
                layer_idx=i,
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0])
        args = self.args
        if args.use_cla:
            cache_for_layer = [
                cache[(i // args.cla_share_factor) * args.cla_share_factor]
                for i in range(len(self.layers))
            ]
        else:
            cache_for_layer = cache

        shared_kv_states = None
        for layer, c in zip(self.layers, cache_for_layer, strict=True):
            h, shared_kv_states = layer(h, mask, c, shared_kv_states)
        return self.norm(h)


# ----- Model (ModelProtocol) -----


class Model(nn.Module):
    """HunYuan LM — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HunYuanModel(args)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        return self.model.embed_tokens.as_linear(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if "model.layers.0.mlp.gate_and_up_proj.weight" in weights:
            new_weights: dict[str, Any] = {}
            D = self.args.hidden_size
            n_kv_heads = self.args.num_key_value_heads
            n_kv_groups = self.args.num_attention_heads // n_kv_heads
            head_dim = D // self.args.num_attention_heads
            for k, v in weights.items():
                if "qkv_proj" in k:
                    v = v.reshape(n_kv_heads, n_kv_groups + 2, head_dim, -1)
                    splits = v.split([n_kv_groups, n_kv_groups + 1], axis=1)
                    for k_up, v_new in zip(
                        ["q_proj", "k_proj", "v_proj"], splits, strict=False
                    ):
                        k_new = k.replace("qkv_proj", k_up)
                        new_weights[k_new] = mx.flatten(v_new, 0, 2)
                elif "gate_and_up_proj" in k:
                    splits = v.split(2, axis=0)
                    for k_up, v_new in zip(
                        ["up_proj", "gate_proj"], splits, strict=False
                    ):
                        k_new = k.replace("gate_and_up_proj", k_up)
                        new_weights[k_new] = v_new
                else:
                    new_weights[k] = v
            weights = new_weights

        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                for k in ["weight", "scales", "biases"]:
                    key = f"{prefix}.mlp.experts.0.{n}.{k}"
                    if key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(
                            to_join
                        )
        return weights

    def parameters(self) -> dict[str, Any]:
        return dict(self.items())
