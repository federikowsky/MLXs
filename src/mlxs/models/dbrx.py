"""DBRX Sparse MoE model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports fused QKV with clip, RoPE, LayerNorm, and sparse mixture-of-experts.
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
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """DBRX model configuration."""

    model_type: str = "dbrx"
    vocab_size: int = 100352
    d_model: int = 6144
    ffn_config: dict | None = None
    attn_config: dict | None = None
    n_layers: int = 40
    n_heads: int = 48

    def __post_init__(self) -> None:
        if self.ffn_config is None:
            self.ffn_config = {}
        if self.attn_config is None:
            self.attn_config = {}


class Attention(nn.Module):
    """Fused QKV attention with clip and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_heads = args.n_heads
        self.d_model = args.d_model
        self.head_dim = args.d_model // args.n_heads
        self.num_key_value_heads = args.attn_config.get("kv_n_heads", args.n_heads)
        self.clip_qkv = args.attn_config.get("clip_qkv", None)
        rope_theta = args.attn_config.get("rope_theta", 10000.0)

        self.scale = self.head_dim**-0.5

        self.Wqkv = nn.Linear(
            args.d_model,
            (self.num_key_value_heads * 2 + self.num_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            base=rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.Wqkv(x)
        if self.clip_qkv is not None:
            qkv = mx.clip(qkv, a_min=-self.clip_qkv, a_max=self.clip_qkv)

        splits = [self.d_model, self.d_model + self.head_dim * self.num_key_value_heads]
        queries, keys, values = mx.split(qkv, splits, axis=-1)

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask,
        )
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class NormAttnNorm(nn.Module):
    """Pre-norm attention with post-norm output for MoE input."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(args.d_model, bias=False)
        self.norm_2 = nn.LayerNorm(args.d_model, bias=False)
        self.attn = Attention(args)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, mx.array]:
        h = self.attn(self.norm_1(x), mask=mask, cache=cache)
        x = h + x
        return x, self.norm_2(x)


class ExpertMLP(nn.Module):
    """Single expert MLP with SwiGLU activation."""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.v1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.v1(x)))


class Router(nn.Module):
    """Expert routing gate."""

    def __init__(self, d_model: int, num_experts: int) -> None:
        super().__init__()
        self.layer = nn.Linear(d_model, num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.layer(x)


class SparseMoeBlock(nn.Module):
    """Sparse MoE block with per-token expert routing."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.d_model = args.d_model
        self.ffn_dim = args.ffn_config.get("ffn_hidden_size", args.d_model * 4)
        self.num_experts = args.ffn_config.get("moe_num_experts", 16)
        self.num_experts_per_tok = args.ffn_config.get("moe_top_k", 4)

        self.router = Router(self.d_model, self.num_experts)
        self.experts = [ExpertMLP(self.d_model, self.ffn_dim) for _ in range(self.num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne - 1, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        y = []
        for xt, st, it in zip(x, scores, inds.tolist()):
            yt = mx.stack([self.experts[e](xt) for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt)
        y = mx.stack(y, axis=0)

        return y.reshape(orig_shape)


class DecoderLayer(nn.Module):
    """DBRX decoder layer: NormAttnNorm + SparseMoE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.ffn = SparseMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r, h = self.norm_attn_norm(x, mask, cache)
        return self.ffn(h) + r


class DBRXModel(nn.Module):
    """DBRX transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args) for _ in range(args.n_layers)]
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.blocks)  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.blocks, cache, strict=True):
            h = layer(h, mask, c)

        return self.norm_f(h)


class Model(nn.Module):
    """DBRX LM head wrapper -- satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = DBRXModel(args)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.transformer(inputs, cache)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.transformer.blocks)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.transformer.blocks]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        num_experts = self.args.ffn_config.get("moe_num_experts", 16)
        dim = self.args.ffn_config.get("ffn_hidden_size", self.args.d_model * 4)

        pattern = "experts.mlp"
        new_weights = {k: v for k, v in weights.items() if pattern not in k}
        for k, v in weights.items():
            if pattern in k:
                experts = [
                    (k.replace(".mlp", f".{e}") + ".weight", sv)
                    for e, sv in enumerate(mx.split(v, num_experts, axis=0))
                ]
                if k.endswith("w2"):
                    experts = [(s, sv.T) for s, sv in experts]
                new_weights.update(experts)
        return new_weights

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.transformer.blocks
