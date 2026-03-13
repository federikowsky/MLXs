"""PLaMo model — port from mlx_lm (mlx_lm/models/plamo.py), ModelProtocol-compliant.

Dense decoder with shared KV heads (n_shared_head): fewer K/V heads, tiled for Q heads.
RoPE, SwiGLU MLP, RMSNorm. No tie_word_embeddings in config.
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
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """PLaMo config (config.json)."""

    model_type: str = "plamo"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    n_shared_head: int = 8
    rope_theta: float = 10000.0
    rope_traditional: bool = False


class Attention(nn.Module):
    """Shared KV heads: k_num_heads = ceil(q_num_heads / n_shared_head), then tile K,V."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.q_num_heads = config.num_attention_heads
        head_dim = dim // self.q_num_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = (
            self.q_num_heads + config.n_shared_head - 1
        ) // config.n_shared_head
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.q_num_heads * self.qk_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.k_num_heads * self.qk_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.v_num_heads * self.v_dim, bias=False)
        self.o_proj = nn.Linear(self.q_num_heads * self.v_dim, dim, bias=False)
        self.rotary_emb = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=1.0,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape
        queries = (
            self.q_proj(hidden_states)
            .reshape(bsz, q_len, self.q_num_heads, self.qk_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(hidden_states)
            .reshape(bsz, q_len, self.k_num_heads, self.qk_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(hidden_states)
            .reshape(bsz, q_len, self.v_num_heads, self.v_dim)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            queries = self.rotary_emb(queries, offset=cache.offset)
            keys = self.rotary_emb(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rotary_emb(queries)
            keys = self.rotary_emb(keys)

        keys = mx.tile(keys, [1, self.config.n_shared_head, 1, 1])
        values = mx.tile(values, [1, self.config.n_shared_head, 1, 1])

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=attention_mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1))


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class PlamoDecoderLayer(nn.Module):
    """Pre-norm for attn, then residual + MLP (no norm before MLP) + residual."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states_sa = self.self_attn(hidden_states, attention_mask, cache)
        hidden_states_mlp = self.mlp(hidden_states)
        return residual + hidden_states_sa + hidden_states_mlp


class PlamoModel(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers_list = [PlamoDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers_list)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers_list, cache, strict=True):
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """PLaMo LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.model = PlamoModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers_list)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers_list]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights
