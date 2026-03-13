"""Nemotron-NAS: NAS-style heterogeneous transformer — port from mlx_lm.

Per-layer block configs: attention (GQA groups, no-op, or linear replacement) and
FFN (ffn_mult, no-op, or linear). RMSNorm, RoPE, optional rope_scaling. ModelProtocol-compliant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs

# ----- Block config (per-layer attention/FFN) -----


@dataclass(frozen=True)
class AttentionConfig:
    """Per-layer attention config: no-op, linear replacement, or GQA with n_heads_in_group."""

    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: list[str] | None = None
    n_heads_in_group: int | None = None
    window_length: int | None = None
    num_sink_tokens: int | None = None
    use_prefill_window_in_sink_attention: bool = False
    unshifted_sink: bool = False

    def __post_init__(self) -> None:
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "n_heads_in_group", None)
        elif not self.no_op and self.n_heads_in_group is None:
            raise ValueError("n_heads_in_group must be specified for active attention blocks")
        elif not self.no_op and (self.n_heads_in_group or 0) <= 0:
            raise ValueError("n_heads_in_group must be positive")


@dataclass(frozen=True)
class FFNConfig:
    """Per-layer FFN config: no-op, linear replacement, or standard with ffn_mult."""

    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: list[str] | None = None
    ffn_mult: float | None = None

    def __post_init__(self) -> None:
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "ffn_mult", None)
        elif not self.no_op and self.ffn_mult is None:
            raise ValueError("ffn_mult must be specified for active FFN blocks")
        elif not self.no_op and self.ffn_mult is not None:
            object.__setattr__(self, "ffn_mult", round(self.ffn_mult, 6))


@dataclass(frozen=True)
class BlockConfig:
    """Attention + FFN config for one layer."""

    attention: AttentionConfig
    ffn: FFNConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockConfig:
        attn = AttentionConfig(**data.get("attention", {}))
        ffn = FFNConfig(**data.get("ffn", {}))
        return cls(attention=attn, ffn=ffn)


def _find_multiple(n: int, k: int) -> int:
    """Smallest multiple of k >= n."""
    if n % k == 0:
        return n
    return n + k - (n % k)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    """Intermediate size from ffn_mult, rounded up to multiple of 256."""
    size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(size, 256)


_ACT2FN: dict[str, Any] = {
    "silu": nn.silu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
    "gelu_fast": nn.gelu_approx,
}


# ----- ModelArgs -----


@dataclass
class ModelArgs(BaseModelArgs):
    """Nemotron-NAS config (config.json)."""

    model_type: str = "nemotron_nas"
    hidden_size: int = 8192
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128256
    block_configs: list[BlockConfig] | list[dict[str, Any]] = field(default_factory=list)
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.block_configs and isinstance(self.block_configs[0], dict):
            self.block_configs = [
                BlockConfig.from_dict(conf)
                for conf in self.block_configs  # type: ignore[union-attr]
            ]
        if self.num_hidden_layers > 0 and len(self.block_configs) != self.num_hidden_layers:
            raise ValueError(
                f"block_configs length ({len(self.block_configs)}) must equal "
                f"num_hidden_layers ({self.num_hidden_layers})"
            )
        if self.rope_scaling:
            if "factor" not in self.rope_scaling:
                raise ValueError("rope_scaling must contain 'factor'")
            if (
                self.rope_scaling.get("rope_type") is None
                and self.rope_scaling.get("type") is None
            ):
                raise ValueError("rope_scaling must contain 'rope_type' or 'type'")
        for i, block_conf in enumerate(self.block_configs):
            attn_conf = block_conf.attention
            n_group = attn_conf.n_heads_in_group
            if (
                not attn_conf.no_op
                and not attn_conf.replace_with_linear
                and n_group is not None
                and self.num_attention_heads % n_group != 0
            ):
                raise ValueError(
                    f"Layer {i}: num_attention_heads ({self.num_attention_heads}) "
                    f"must be divisible by n_heads_in_group ({n_group})"
                )


# ----- Attention -----


class Attention(nn.Module):
    """GQA attention for layers that use it; RoPE from config."""

    def __init__(self, args: ModelArgs, attention_config: AttentionConfig) -> None:
        super().__init__()
        dim = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv_heads = n_heads // (attention_config.n_heads_in_group or 1)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim = dim // n_heads
        if head_dim * n_heads != dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

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
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# ----- MLP -----


class MLP(nn.Module):
    """FFN with gate * up and configurable activation (silu, relu, gelu, etc.)."""

    def __init__(self, args: ModelArgs, ffn_config: FFNConfig) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden_dim = _ffn_mult_to_intermediate_size(ffn_config.ffn_mult or 0.0, dim)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        if args.hidden_act not in _ACT2FN:
            raise ValueError(f"Unknown hidden_act: {args.hidden_act}")
        self._act_fn = args.hidden_act

    def __call__(self, x: mx.array) -> mx.array:
        act = _ACT2FN[self._act_fn]
        return self.down_proj(act(self.gate_proj(x)) * self.up_proj(x))


# ----- Linear replacement (no-op / replace subblocks) -----


class LinearSubblockReplacement(nn.Module):
    """Single linear layer replacing attention or MLP subblock."""

    def __init__(self, hidden_size: int, bias: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        return self.linear(x)


# ----- Transformer block -----


class TransformerBlock(nn.Module):
    """One transformer block: optional pre-norm + attention, optional pre-norm + MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        block_config = args.block_configs[layer_idx]
        self.attention_config = block_config.attention
        self.ffn_config = block_config.ffn

        if not self.attention_config.no_op:
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        else:
            self.input_layernorm = None

        if self.attention_config.no_op:
            self.self_attn: Attention | LinearSubblockReplacement | None = None
        elif self.attention_config.replace_with_linear:
            self.self_attn = LinearSubblockReplacement(args.hidden_size, args.attention_bias)
        else:
            self.self_attn = Attention(args, self.attention_config)

        if not self.ffn_config.no_op:
            self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        else:
            self.post_attention_layernorm = None

        if self.ffn_config.no_op:
            self.mlp: MLP | LinearSubblockReplacement | None = None
        elif self.ffn_config.replace_with_linear:
            self.mlp = LinearSubblockReplacement(args.hidden_size, args.mlp_bias)
        else:
            self.mlp = MLP(args, self.ffn_config)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        if self.self_attn is not None:
            residual = x
            h = self.input_layernorm(x)  # type: ignore[union-attr]
            x = residual + self.self_attn(h, mask=mask, cache=cache)
        if self.mlp is not None:
            residual = x
            h = self.post_attention_layernorm(x)  # type: ignore[union-attr]
            x = residual + self.mlp(h)
        return x


# ----- Backbone -----


class NemotronNASModel(nn.Module):
    """Core Nemotron-NAS transformer: embed + heterogeneous layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._num_attn_layers = sum(1 for layer in self.layers if layer.self_attn is not None)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * self._num_attn_layers  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        cache_idx = 0
        for layer in self.layers:
            c = cache[cache_idx] if layer.self_attn is not None else None
            if layer.self_attn is not None:
                cache_idx += 1
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


# ----- Model (ModelProtocol) -----


class Model(nn.Module):
    """Nemotron-NAS LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = NemotronNASModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.backbone(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.backbone.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)  # type: ignore[union-attr]
        return out

    @property
    def num_layers(self) -> int:
        return len(self.backbone.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for layer in self.backbone.layers if layer.self_attn is not None]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def parameters(self) -> dict[str, Any]:
        return dict(self.items())
