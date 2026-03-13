"""PLaMo 2 hybrid model (Mamba + attention) — port from mlx_lm, ModelProtocol-compliant.

Alternating Mamba and attention layers (mamba_step), RMSNorm with offset,
RoPE + QK-norm attention, SwiGLU MLP. Mamba layers use ArraysCache(size=2);
attention layers use KVCache. Optional tie_word_embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.ssm import ssm_update
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """PLaMo 2 config (config.json)."""

    model_type: str = "plamo2"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_size_per_head: int = 128
    max_position_embeddings: int = 2048
    attention_window_size: int = 2048
    full_attention_idx: list[int] | None = None
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_num_heads: int = 64
    mamba_step: int = 2
    mamba_chunk_size: int = 256
    mamba_enabled: bool = True
    intermediate_size: int = 13312
    vocab_size: int = 32000


class Plamo2RMSNorm(nn.Module):
    """RMSNorm with offset: scale = weight + offset (model-specific, not shared)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        offset: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = mx.zeros(hidden_size)
        self.variance_epsilon = eps
        self.offset = offset

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.fast.rms_norm(hidden_states, self.weight + self.offset, self.variance_epsilon)


class Mamba(nn.Module):
    """Mamba SSM block: conv + BCdt + ssm_update, then SwiGLU with z."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.d_state = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.chunk_size = config.mamba_chunk_size
        self.num_heads = config.mamba_num_heads
        self.hidden_size_per_head = config.hidden_size_per_head
        self.intermediate_size = self.num_heads * self.hidden_size_per_head
        self.dt_dim = max(64, self.hidden_size // 16)

        self.in_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=0,
        )
        self.bcdt_proj = nn.Linear(
            self.intermediate_size,
            self.dt_dim + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_dim, self.num_heads, bias=False)
        self.dt_bias = mx.zeros((self.num_heads,))
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)
        self.dt_norm_weight = mx.ones(self.dt_dim)
        self.B_norm_weight = mx.ones(self.d_state)
        self.C_norm_weight = mx.ones(self.d_state)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def _conv(
        self,
        conv_input: mx.array,
        cache: ArraysCache | None,
        mask: mx.array | None,
    ) -> mx.array:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)
        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (
                        conv_input.shape[0],
                        self.conv_kernel_size - 1,
                        self.intermediate_size,
                    ),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)])
        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        x: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: ArraysCache | None,
        mask: mx.array | None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.hidden_size_per_head)
        B = B.reshape(batch_size, seq_len, 1, self.d_state)
        C = C.reshape(batch_size, seq_len, 1, self.d_state)
        state = cache[1] if cache is not None else None
        lengths = cache.lengths if cache is not None else None
        y, state = ssm_update(
            x,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            state,
            time_step_limit=(0.001, 100.0),
            mask=mask,
            lengths=lengths,
        )
        if cache is not None:
            cache[1] = state
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        bsize, length, _ = hidden_states.shape
        zx = self.in_proj(hidden_states)
        zx = zx.reshape(bsize, length, self.num_heads, -1)
        z, x = mx.split(zx, [self.hidden_size_per_head], axis=-1)
        x = x.reshape(bsize, -1, self.num_heads * self.hidden_size_per_head)
        x = self._conv(x, cache, mask)
        BCdt = self.bcdt_proj(x)
        B, C, dt = mx.split(BCdt, [self.d_state, self.d_state * 2], axis=-1)
        dt = mx.fast.rms_norm(dt, self.dt_norm_weight, self.config.rms_norm_eps)
        B = mx.fast.rms_norm(B, self.B_norm_weight, self.config.rms_norm_eps)
        C = mx.fast.rms_norm(C, self.C_norm_weight, self.config.rms_norm_eps)
        dt = self.dt_proj(dt)
        out = self._ssm(x, B, C, dt, cache, mask)
        if cache is not None:
            cache.advance(out.shape[1])
        out = swiglu(z.reshape(bsize, length, -1), out)
        return self.out_proj(out)


class Attention(nn.Module):
    """Multi-head attention with RoPE and QK-norm (RMSNorm then learned scale)."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = config.hidden_size_per_head
        self.scale = head_dim**-0.5
        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = config.num_key_value_heads
        self.n_group = self.q_num_heads // self.k_num_heads
        q_proj_dim = self.q_num_heads * self.qk_dim
        k_proj_dim = self.k_num_heads * self.qk_dim
        v_proj_dim = self.k_num_heads * self.v_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            q_proj_dim + k_proj_dim + v_proj_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.q_num_heads * self.v_dim, self.hidden_size, bias=False)
        self.q_weight = mx.ones((self.q_num_heads, self.qk_dim))
        self.k_weight = mx.ones((self.k_num_heads, self.qk_dim))
        self.rope = nn.RoPE(self.qk_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, T, _ = hidden_states.shape
        q_proj_dim = self.q_num_heads * self.qk_dim
        k_proj_dim = self.k_num_heads * self.qk_dim
        qkv = self.qkv_proj(hidden_states)
        q, k, v = mx.split(qkv, [q_proj_dim, q_proj_dim + k_proj_dim], axis=-1)
        q = q.reshape(B, T, self.q_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.k_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.v_num_heads, self.v_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rms_norm(q, None, 1e-6) * self.q_weight[:, None]
        k = mx.fast.rms_norm(k, None, 1e-6) * self.k_weight[:, None]
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)
        output = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, T, self.q_num_heads * self.v_dim)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU FFN."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.gate_up_proj(x)
        gate, up = mx.split(h, 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


def _is_mamba_layer(config: ModelArgs, layer_idx: int) -> bool:
    if not config.mamba_enabled:
        return False
    if config.num_hidden_layers <= (config.mamba_step // 2):
        return layer_idx != config.num_hidden_layers - 1
    return (layer_idx % config.mamba_step) != (config.mamba_step // 2)


class Plamo2DecoderLayer(nn.Module):
    """Single block: mixer (Mamba or Attention) + MLP, each with pre/post RMSNorm+offset."""

    def __init__(self, config: ModelArgs, is_mamba: bool) -> None:
        super().__init__()
        self.config = config
        self.is_mamba = is_mamba
        if is_mamba:
            self.mixer = Mamba(config)
        else:
            self.mixer = Attention(config)
        self.mlp = MLP(config)
        self.pre_mixer_norm = Plamo2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mixer_norm = Plamo2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / 5
        )
        self.pre_mlp_norm = Plamo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0)
        self.post_mlp_norm = Plamo2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / (5**1.5)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.pre_mixer_norm(hidden_states)
        if self.is_mamba:
            hidden_states_sa = self.mixer(
                hidden_states,
                mask=cast(mx.array | None, mask),
                cache=cast(ArraysCache | None, cache),
            )
        else:
            hidden_states_sa = self.mixer(
                hidden_states, mask=mask, cache=cast(KVCache | None, cache)
            )
        hidden_states_sa = self.post_mixer_norm(hidden_states_sa)
        hidden_states = residual + hidden_states_sa
        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states_mlp = self.post_mlp_norm(hidden_states_mlp)
        return residual + hidden_states_mlp


class Plamo2Decoder(nn.Module):
    """Stack of Plamo2DecoderLayer; attn mask from first attn layer, SSM mask from first Mamba."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.layers = [
            Plamo2DecoderLayer(config, _is_mamba_layer(config, i))
            for i in range(config.num_hidden_layers)
        ]
        self._ssm_idx: int | None = 0 if config.mamba_enabled else None
        self._fa_idx = config.mamba_step // 2

    def __call__(
        self,
        x: mx.array,
        cache: list[KVCache | ArraysCache] | None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        attn_mask = create_attention_mask(x, cache[self._fa_idx])
        mamba_mask = (
            create_ssm_mask(x, cache[self._ssm_idx]) if self._ssm_idx is not None else None
        )
        for layer, c in zip(self.layers, cache, strict=True):
            mask = mamba_mask if layer.is_mamba else attn_mask
            x = layer(x, mask=mask, cache=c)
        return x


class Plamo2Model(nn.Module):
    """Embed + decoder + final RMSNorm."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers_module = Plamo2Decoder(config)
        self.norm = Plamo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        return self.norm(self.layers_module(h, cache))


class Model(nn.Module):
    """PLaMo 2 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.model = Plamo2Model(args)
        self._layer_is_mamba = [_is_mamba_layer(args, i) for i in range(args.num_hidden_layers)]
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | ArraysCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers_module.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return [ArraysCache(size=2) if is_mb else KVCache() for is_mb in self._layer_is_mamba]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights
