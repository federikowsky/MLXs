"""Jamba hybrid model (Mamba + attention + MoE) — port from mlx_lm, ModelProtocol-compliant.

Architecture: alternating Mamba and attention layers (configurable via attn_layer_*),
with dense or MoE FFN per layer. Each layer has a single cache: KVCache for attention
layers, ArraysCache(size=2) for Mamba (conv + SSM state). Masks use the first attn
and first mamba layer caches (mlx_lm create_attention_mask(h, cache[attn_idx]) pattern).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Jamba config; from_dict maps config.json (same field names as mlx_lm/HF)."""

    model_type: str = "jamba"
    hidden_size: int = 4096
    intermediate_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    attn_layer_offset: int = 0
    attn_layer_period: int = 2
    expert_layer_offset: int = 0
    expert_layer_period: int = 2
    mamba_d_conv: int = 4
    mamba_d_state: int = 16
    mamba_expand: int = 2
    num_experts: int = 1
    num_experts_per_tok: int = 1
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 262144
    vocab_size: int = 128256
    mamba_dt_rank: int | str = "auto"
    mamba_proj_bias: bool = False
    mamba_conv_bias: bool = True
    layers_block_type: list[str] | None = field(default=None, repr=False)
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.mamba_dt_rank == "auto":
            self.mamba_dt_rank = math.ceil(self.hidden_size / 16)
        else:
            self.mamba_dt_rank = int(self.mamba_dt_rank)
        if self.layers_block_type is None:
            self.layers_block_type = [
                "attention" if i % self.attn_layer_period == self.attn_layer_offset else "mamba"
                for i in range(self.num_hidden_layers)
            ]


class JambaMLP(nn.Module):
    """Dense FFN (SwiGLU). mlx_lm: JambaMLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class JambaAttention(nn.Module):
    """Multi-head attention without RoPE. mlx_lm: JambaAttention."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@mx.compile
def _fma(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    return a * b + c


class JambaMambaMixer(nn.Module):
    """Mamba block with dt/B/C layernorms. mlx_lm: JambaMambaMixer."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_expand * args.hidden_size
        self.time_step_rank = args.mamba_dt_rank
        self.use_conv_bias = args.mamba_conv_bias
        self.use_bias = args.mamba_proj_bias

        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            bias=self.use_conv_bias,
            padding=0,
        )
        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        A = mx.repeat(
            mx.arange(1.0, self.ssm_state_size + 1.0).reshape((1, self.ssm_state_size)),
            repeats=self.intermediate_size,
            axis=0,
        )
        self.A_log = mx.log(A)
        self.D = mx.ones((self.intermediate_size,))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)
        self.dt_layernorm = nn.RMSNorm(self.time_step_rank, eps=args.rms_norm_eps)
        self.b_layernorm = nn.RMSNorm(self.ssm_state_size, eps=args.rms_norm_eps)
        self.c_layernorm = nn.RMSNorm(self.ssm_state_size, eps=args.rms_norm_eps)

    def _ssm_step(
        self,
        x: mx.array,
        A: mx.array,
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Single-step SSM; returns (y, new_state)."""
        delta_bc = self.x_proj(x)
        delta, B, C = mx.split(
            delta_bc,
            [self.time_step_rank, self.time_step_rank + self.ssm_state_size],
            axis=-1,
        )
        delta = self.dt_layernorm(delta)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, -2)
        dt_a = mx.exp(mx.expand_dims(delta, -1) * A)
        if state is not None:
            new_state = _fma(state, dt_a, new_state)
        y = (new_state @ mx.expand_dims(C, -1)).squeeze(-1)
        y = y + self.D * x
        return y, new_state

    def _process_sequence(
        self,
        x: mx.array,
        conv_state: mx.array | None,
        ssm_state: mx.array | None,
    ) -> tuple[mx.array, tuple[mx.array | None, mx.array | None]]:
        xz = self.in_proj(x)
        x, z = mx.split(xz, 2, axis=-1)
        K = self.conv_kernel_size
        if conv_state is not None:
            x_full = mx.concatenate([conv_state, x], axis=1)
        else:
            x_full = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        conv_out = self.conv1d(x_full)
        new_conv_state = x_full[:, -(K - 1) :, :]
        x = nn.silu(conv_out)
        A = -mx.exp(self.A_log)
        _B, T, _ = x.shape
        current_state = ssm_state
        y_list: list[mx.array] = []
        for t in range(T):
            y_t, current_state = self._ssm_step(x[:, t], A, current_state)
            y_list.append(y_t)
        y = mx.stack(y_list, axis=1)
        z = self.out_proj(swiglu(z, y))
        return z, (new_conv_state, current_state)

    def __call__(
        self,
        x: mx.array,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        if cache is None:
            conv_state, ssm_state = None, None
        else:
            conv_state, ssm_state = cache[0], cache[1]
        output, (new_conv_state, new_ssm_state) = self._process_sequence(x, conv_state, ssm_state)
        if cache is not None:
            cache[0] = new_conv_state
            cache[1] = new_ssm_state
        return output


class JambaSparseMoeBlock(nn.Module):
    """Router + SwitchGLU experts. mlx_lm: JambaSparseMoeBlock."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.router = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.intermediate_size, args.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.router(x)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class JambaDecoderLayer(nn.Module):
    """Single block: attn or mamba + FFN (MLP or MoE). mlx_lm: JambaDecoderLayer."""

    def __init__(self, args: ModelArgs, layer_type: str, layer_idx: int) -> None:
        super().__init__()
        self.is_attn = layer_type == "attention"
        if self.is_attn:
            self.self_attn = JambaAttention(args)
        else:
            self.mamba = JambaMambaMixer(args)
        use_moe = (
            args.num_experts > 1
            and (layer_idx + args.expert_layer_offset) % args.expert_layer_period == 0
        )
        self.feed_forward = JambaSparseMoeBlock(args) if use_moe else JambaMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        if self.is_attn:
            h = self.self_attn(h, mask=mask, cache=cache)
        else:
            h = self.mamba(h, cache=cache)
        r = x + h
        return r + self.feed_forward(self.pre_ff_layernorm(r))


class JambaModel(nn.Module):
    """Embed + layers + final norm. mlx_lm: JambaModel."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            JambaDecoderLayer(args, t, idx) for idx, t in enumerate(args.layers_block_type)
        ]
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._attn_idx = args.layers_block_type.index("attention")
        self._ssm_idx = args.layers_block_type.index("mamba")

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        attn_mask = create_attention_mask(h, cache[self._attn_idx])
        ssm_mask = create_ssm_mask(h, cache[self._ssm_idx])
        for layer, c in zip(self.layers, cache, strict=True):
            mask = attn_mask if layer.is_attn else ssm_mask
            h = layer(h, mask=mask, cache=c)
        return self.final_layernorm(h)


class Model(nn.Module):
    """Top-level Jamba; implements ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.model = JambaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | ArraysCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.model(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    def make_cache(self) -> list[KVCache | ArraysCache]:
        caches: list[KVCache | ArraysCache] = []
        for layer in self.model.layers:
            if layer.is_attn:
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=2))
        return caches

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Conv1d weight transpose, pop tied lm_head, stack experts → switch_mlp."""
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        num_experts = self.args.num_experts
        for layer_idx in range(self.args.num_hidden_layers):
            base = f"model.layers.{layer_idx}.feed_forward"
            if not any(key.startswith(f"{base}.experts.") for key in weights):
                continue
            for proj in ["gate_proj", "down_proj", "up_proj"]:
                for name in ["weight", "bias", "scales", "biases"]:
                    expert_tensors = [
                        weights.pop(f"{base}.experts.{e}.{proj}.{name}")
                        for e in range(num_experts)
                        if f"{base}.experts.{e}.{proj}.{name}" in weights
                    ]
                    if expert_tensors:
                        weights[f"{base}.switch_mlp.{proj}.{name}"] = mx.stack(expert_tensors)
        return weights
