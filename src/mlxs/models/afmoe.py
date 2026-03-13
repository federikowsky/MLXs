"""AFMoE model: full + sliding attention, dense + MoE layers, gated attention (§7).

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Uses RMSNorm (nn.RMSNorm), RoPE only on sliding layers, SwitchGLU experts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """AFMoE model configuration."""

    model_type: str = "afmoe"
    layer_types: list[str] = ()
    vocab_size: int = 200192
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    num_dense_layers: int = 2
    route_norm: bool = True
    route_scale: float = 2.826
    score_func: str = "sigmoid"
    n_group: int = 1
    topk_group: int = 1
    sliding_window: int = 2048
    mup_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


class Attention(nn.Module):
    """Multi-head attention with optional Q/K norm, RoPE (sliding only), and output gate."""

    def __init__(self, args: ModelArgs, is_local_attention: bool = False) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.is_local_attention = is_local_attention
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.gate_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)

        if is_local_attention:
            self.rope = initialize_rope(
                self.head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )
        else:
            self.rope = None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if self.is_local_attention and self.rope is not None:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gate = mx.sigmoid(self.gate_proj(x))
        output = output * gate
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, args: ModelArgs, intermediate_size: int | None = None) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size if intermediate_size is not None else args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoERouter(nn.Module):
    """Router gate for MoE expert selection."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.gate(x)


class AfmoeMoE(nn.Module):
    """MoE block with router, expert bias, optional shared experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.route_norm = args.route_norm
        self.route_scale = args.route_scale
        self.score_func = args.score_func
        self.n_group = args.n_group
        self.topk_group = args.topk_group

        self.router = MoERouter(args)
        self.expert_bias = mx.zeros((args.num_experts,))
        self.experts = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
        if args.num_shared_experts > 0:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = MLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.router(x)
        if self.score_func == "sigmoid":
            scores = mx.sigmoid(gates.astype(mx.float32))
        else:
            scores = mx.softmax(gates.astype(mx.float32), axis=-1)
        selection_scores = scores + self.expert_bias

        if self.n_group > 1:
            selection_scores = mx.unflatten(selection_scores, axis=-1, shape=(self.n_group, -1))
            group_scores = mx.topk(selection_scores, 2, axis=-1).sum(axis=-1, keepdims=True)
            k = self.n_group - self.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            selection_scores = mx.put_along_axis(
                selection_scores,
                mx.stop_gradient(group_idx),
                mx.array(0.0),
                axis=-2,
            )
            selection_scores = mx.flatten(selection_scores, -2, -1)

        k = self.num_experts_per_tok
        inds = mx.argpartition(-selection_scores, kth=k - 1, axis=-1)[..., :k]
        selected_scores = mx.take_along_axis(scores, inds, axis=-1)

        if self.route_norm and self.num_experts_per_tok > 1:
            denominator = selected_scores.sum(axis=-1, keepdims=True)
            selected_scores = selected_scores / denominator
        selected_scores = selected_scores * self.route_scale

        y = self.experts(x, inds)
        y = (y * selected_scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    """Pre-norm attention + pre-norm MLP or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int, use_sliding: bool = False) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.self_attn = Attention(args, is_local_attention=use_sliding)
        if layer_idx < args.num_dense_layers:
            self.mlp = MLP(args)
        else:
            self.mlp = AfmoeMoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        r = self.post_attention_layernorm(r)
        h = x + r
        r = self.mlp(self.pre_mlp_layernorm(h))
        r = self.post_mlp_layernorm(r)
        return h + r


class AfmoeModel(nn.Module):
    """Transformer stack: embed, decoder layers, final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.mup_enabled = args.mup_enabled
        self.hidden_size = args.hidden_size

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args,
                layer_idx=idx,
                use_sliding=layer_type == "sliding_attention",
            )
            for idx, layer_type in enumerate(self.layer_types)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self._fa_idx = self.layer_types.index("full_attention")
        self._swa_idx: int | None = None
        for idx, layer in enumerate(self.layers):
            if layer.use_sliding:
                self._swa_idx = idx
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if self.mup_enabled:
            h = h * math.sqrt(self.hidden_size)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        fa_mask = create_attention_mask(h, cache[self._fa_idx])
        swa_mask: mx.array | str | None = None
        if self._swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self._swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache, strict=True):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """AFMoE model (ModelProtocol): embed + decoder + lm_head or tied embed."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AfmoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)  # type: ignore[union-attr]
        return out

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.model.layers
        ]

    @property
    def num_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for layer_idx in range(self.args.num_hidden_layers):
            if layer_idx < self.args.num_dense_layers:
                continue
            prefix = f"model.layers.{layer_idx}"
            for n in ("up_proj", "down_proj", "gate_proj"):
                for k in ("weight", "scales", "biases"):
                    key = f"{prefix}.mlp.experts.0.{n}.{k}"
                    if key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.experts.{n}.{k}"] = mx.stack(to_join)
        return weights

    def parameters(self) -> dict[str, Any]:
        return dict(self.items())
