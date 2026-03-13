"""GraniteMoE model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports GQA, RoPE (with scaling), residual/embedding/logits scaling,
and Mixture-of-Experts with SwitchGLU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache
from mlxs.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlxs.models.rope import initialize_rope
from mlxs.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    """GraniteMoE model configuration."""

    model_type: str = "granitemoe"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    logits_scaling: float = 1.0
    attention_multiplier: float = 1.0
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    max_position_embeddings: int = 4096
    num_key_value_heads: int = 8
    attention_bias: bool = False
    rope_theta: float = 10000.0
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = True


class GraniteMoeAttention(nn.Module):
    """Multi-head attention with GQA, RoPE, and attention scaling."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = args.attention_multiplier
        bias = args.attention_bias

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class GraniteMoeTopKGating(nn.Module):
    """Top-k gating for GraniteMoE."""

    def __init__(self, input_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.layer(hidden_states)
        top_k_idx = mx.argpartition(logits, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        top_k_logits = mx.take_along_axis(logits, top_k_idx, axis=-1)
        top_k_gates = mx.softmax(top_k_logits.astype(mx.float32), axis=-1)
        return top_k_idx, top_k_gates


class GraniteMoeMoE(nn.Module):
    """Mixture-of-Experts block with top-k gating and SwitchGLU."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.input_size = args.hidden_size
        self.hidden_size = args.intermediate_size
        self.switch_mlp = SwitchGLU(
            self.input_size, self.hidden_size, args.num_local_experts,
        )
        self.router = GraniteMoeTopKGating(
            input_size=self.input_size,
            num_experts=args.num_local_experts,
            top_k=args.num_experts_per_tok,
        )

    def __call__(self, x: mx.array) -> mx.array:
        token_ids, gates = self.router(x)
        y = self.switch_mlp(x, token_ids)
        return (y * gates[..., None]).sum(axis=-2).astype(y.dtype)


class GraniteMoeDecoderLayer(nn.Module):
    """Pre-norm transformer block with MoE MLP and residual scaling."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = GraniteMoeAttention(args)
        self.block_sparse_moe = GraniteMoeMoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps,
        )
        self.residual_multiplier = args.residual_multiplier

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r * self.residual_multiplier
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        return h + r * self.residual_multiplier


class GraniteMoEModel(nn.Module):
    """GraniteMoE transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            GraniteMoeDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) * self.embedding_multiplier
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """GraniteMoE LM head wrapper — satisfies ModelProtocol (AC17)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteMoEModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out / self.logits_scaling

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if "model.layers.0.block_sparse_moe.input_linear.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.block_sparse_moe"
            key = f"{prefix}.input_linear.weight"
            value = weights.pop(key)
            gate_proj, up_proj = mx.split(value, 2, axis=1)
            weights[key.replace("input_linear", "switch_mlp.gate_proj")] = gate_proj
            weights[key.replace("input_linear", "switch_mlp.up_proj")] = up_proj
            key = f"{prefix}.output_linear.weight"
            weights[key.replace("output_linear", "switch_mlp.down_proj")] = (
                weights.pop(key)
            )
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def quant_predicate(self) -> Any:
        def predicate(path: str, _: Any) -> dict[str, int] | bool:
            if path.endswith("block_sparse_moe.router.layer"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self) -> list[GraniteMoeDecoderLayer]:
        return self.model.layers
