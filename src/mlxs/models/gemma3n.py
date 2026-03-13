"""Gemma 3N model — port from mlx_lm (mlx_lm/models/gemma3n.py).

Distinct from gemma3: AltUp (Alternating Updates), Laurel (Learned Augmented
Residual Layer), per-layer inputs, KV-shared layers, optional activation
sparsity and final logit softcapping. Implements ModelProtocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.models.base import BaseModelArgs


@dataclass
class TextConfig(BaseModelArgs):
    """Gemma 3N text config; from_dict aligned to config.json text_config."""

    model_type: str = "gemma3n"
    hidden_size: int = 2048
    num_hidden_layers: int = 46
    intermediate_size: int | list[int] = 8192
    num_attention_heads: int = 16
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 256128
    num_key_value_heads: int = 8
    num_kv_shared_layers: int = 8
    vocab_size_per_layer_input: int = 256
    sliding_window: int = 4096
    max_position_embeddings: int = 131072
    rope_local_base_freq: float = 10_000.0
    rope_theta: float = 1_000_000.0
    final_logit_softcapping: float | None = None
    layer_types: list[str] | None = None
    activation_sparsity_pattern: list[float] | None = None
    hidden_size_per_layer_input: int = 256
    altup_num_inputs: int = 2
    altup_coef_clip: float | None = None
    altup_correct_scale: bool = False
    altup_active_idx: int = 0
    laurel_rank: int = 256
    rope_scaling: dict[str, Any] | None = None


@dataclass
class ModelArgs(BaseModelArgs):
    """Top-level config with model_type and nested text_config."""

    model_type: str = "gemma3n"
    text_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.text_config is None:
            self.text_config = {}


class RMSNoScale(nn.Module):
    """RMSNorm without learned scale (gemma3n V-norm)."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class LaurelBlock(nn.Module):
    """Learned Augmented Residual Layer (Laurel)."""

    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.linear_left = nn.Linear(config.hidden_size, config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(config.laurel_rank, config.hidden_size, bias=False)
        self.post_laurel_norm = nn.RMSNorm(dims=config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        laurel_x = self.linear_right(self.linear_left(x))
        return x + self.post_laurel_norm(laurel_x)


class Gemma3nAttention(nn.Module):
    """Attention with Q/K/V norms and optional KV sharing."""

    def __init__(
        self,
        config: TextConfig,
        layer_idx: int,
        is_kv_shared_layer: bool,
    ) -> None:
        super().__init__()
        layer_types = config.layer_types or []
        self.is_sliding = (
            layer_types[layer_idx] == "sliding_attention"
            if layer_idx < len(layer_types)
            else False
        )
        dim = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scale = 1.0
        self.is_kv_shared_layer = is_kv_shared_layer

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(dims=head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNoScale(eps=config.rms_norm_eps)
        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=(config.rope_local_base_freq if self.is_sliding else config.rope_theta),
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, -1, self.head_dim)
        queries = self.q_norm(queries)

        if self.is_kv_shared_layer and cache is not None:
            state = cache.state
            if state is not None:
                keys, values = state
                offset = cache.offset
            else:
                keys = self.k_proj(x).reshape(B, L, -1, self.head_dim)
                keys = self.k_norm(keys).transpose(0, 2, 1, 3)
                keys = self.rope(keys, offset=0)
                values = self.v_proj(x).reshape(B, L, -1, self.head_dim)
                values = self.v_norm(values).transpose(0, 2, 1, 3)
                offset = 0
        else:
            offset = cache.offset if cache is not None else 0
            keys = self.k_proj(x).reshape(B, L, -1, self.head_dim)
            keys = self.k_norm(keys).transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)
            values = self.v_proj(x).reshape(B, L, -1, self.head_dim)
            values = self.v_norm(values).transpose(0, 2, 1, 3)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@partial(mx.compile, shapeless=True)
def _gelu_topk(inputs: mx.array, std_multiplier: mx.array) -> mx.array:
    """GELU with top-k style gating via mean + std cutoff."""
    inputs_mean = mx.mean(inputs, axis=-1, keepdims=True)
    inputs_std = mx.std(inputs, axis=-1, keepdims=True)
    cutoff_x = inputs_mean + inputs_std * std_multiplier.astype(inputs_std.dtype)
    return nn.gelu_approx(mx.maximum(0, inputs - cutoff_x))


class MLP(nn.Module):
    """Gated MLP with optional activation sparsity (gelu_topk)."""

    def __init__(self, config: TextConfig, layer_idx: int = 0) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = (
            config.intermediate_size[layer_idx]
            if isinstance(config.intermediate_size, list)
            else config.intermediate_size
        )
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        pattern = config.activation_sparsity_pattern
        self.activation_sparsity = (
            pattern[layer_idx] if pattern and layer_idx < len(pattern) else 0.0
        )
        if self.activation_sparsity > 0:
            self._std_multiplier = math.sqrt(2.0) * mx.erfinv(2 * self.activation_sparsity - 1)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        activations = (
            _gelu_topk(gate, self._std_multiplier)
            if self.activation_sparsity > 0
            else nn.gelu_approx(gate)
        )
        return self.down_proj(activations * self.up_proj(x))


class AltUp(nn.Module):
    """Alternating Updates (AltUp) block."""

    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.config = config
        self.correct_output_scale = mx.zeros((config.hidden_size,))
        n = config.altup_num_inputs
        self.correction_coefs = nn.Linear(n, n, bias=False)
        self.prediction_coefs = nn.Linear(n, n**2, bias=False)
        self.modality_router = nn.Linear(config.hidden_size, n, bias=False)
        self.router_norm = nn.RMSNorm(dims=config.hidden_size, eps=config.rms_norm_eps)

    def _router_modalities(self, x: mx.array) -> mx.array:
        routed = self.modality_router(
            self.router_norm(x) * (self.config.hidden_size**-1.0)
        ).astype(mx.float32)
        return mx.tanh(routed)

    def predict(self, x: mx.array) -> mx.array:
        modalities = self._router_modalities(x[self.config.altup_active_idx])
        w = self.prediction_coefs.weight.astype(mx.float32)
        if self.config.altup_coef_clip is not None:
            w = mx.clip(
                w,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )
        self.prediction_coefs.weight = w
        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, -1)
            .transpose(0, 1, 3, 2)
        )
        x_up = x.astype(mx.float32)
        x_perm = x_up.transpose(1, 2, 3, 0)
        predictions = mx.matmul(x_perm, all_coefs).transpose(3, 0, 1, 2)
        return (predictions + x_up).astype(x.dtype)

    def correct(self, predictions: mx.array, activated: mx.array) -> mx.array:
        modalities = self._router_modalities(activated)
        w = self.correction_coefs.weight.astype(mx.float32)
        if self.config.altup_coef_clip is not None:
            w = mx.clip(
                w,
                -self.config.altup_coef_clip,
                self.config.altup_coef_clip,
            )
        self.correction_coefs.weight = w
        all_coefs = self.correction_coefs(modalities) + 1.0
        innovation = activated - predictions[self.config.altup_active_idx]
        corrected = innovation[None] * all_coefs.moveaxis(2, 0)[..., None]
        return (corrected + predictions).astype(activated.dtype)


class DecoderLayer(nn.Module):
    """Gemma 3N decoder layer: AltUp, attention, Laurel, MLP, per-layer input."""

    def __init__(
        self,
        config: TextConfig,
        layer_idx: int,
        is_kv_shared_layer: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.self_attn = Gemma3nAttention(config, layer_idx, is_kv_shared_layer)
        self.mlp = MLP(config, layer_idx=layer_idx)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.altup = AltUp(config)
        self.laurel = LaurelBlock(config)
        h_in = config.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(config.hidden_size, h_in, bias=False)
        self.per_layer_projection = nn.Linear(h_in, config.hidden_size, bias=False)
        self.post_per_layer_input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
        per_layer_input: mx.array | None = None,
    ) -> mx.array:
        predictions = self.altup.predict(x)
        active = predictions[self.config.altup_active_idx]
        active_normed = self.input_layernorm(active)
        laurel_out = self.laurel(active_normed)
        attn = self.self_attn(active_normed, mask, cache)
        attn = self.post_attention_layernorm(attn)
        attn_gated = active + attn
        attn_laurel = (attn_gated + laurel_out) * (2.0**-0.5)
        attn_ffw = self.mlp(self.pre_feedforward_layernorm(attn_laurel))
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        corrected = self.altup.correct(predictions, attn_laurel + attn_ffw_norm)
        first = corrected[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            first = first * self.altup.correct_output_scale
        first = mx.multiply(
            nn.gelu_approx(self.per_layer_input_gate(first)),
            per_layer_input,
        )
        first = self.post_per_layer_input_norm(self.per_layer_projection(first))
        corrected = mx.concatenate(
            [
                corrected[: self.config.altup_active_idx + 1],
                corrected[self.config.altup_active_idx + 1 :] + first,
            ],
            axis=0,
        )
        return corrected


@partial(mx.compile, shapeless=True)
def _logit_softcap(softcap: float, x: mx.array) -> mx.array:
    return mx.tanh(x / softcap) * softcap


class Gemma3nBackbone(nn.Module):
    """Gemma 3N language model backbone with AltUp, Laurel, per-layer inputs."""

    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        layer_types = config.layer_types or []

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(
                config=config,
                layer_idx=i,
                is_kv_shared_layer=(i >= self.first_kv_shared_layer_idx),
            )
            for i in range(config.num_hidden_layers)
        ]
        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
        )
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = nn.RMSNorm(
            dims=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )
        self.altup_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(1, config.altup_num_inputs)
        ]
        self.altup_unembed_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(1, config.altup_num_inputs)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.first_sliding_idx = (
            layer_types.index("sliding_attention") if "sliding_attention" in layer_types else 0
        )
        self.first_full_idx = (
            layer_types.index("full_attention") if "full_attention" in layer_types else 0
        )
        self.sliding_window = config.sliding_window

        concrete = layer_types[: self.first_kv_shared_layer_idx]
        self.layer_idx_to_cache_idx = list(range(self.first_kv_shared_layer_idx))
        if self.first_kv_shared_layer_idx < len(layer_types):
            shared_full = len(concrete) - 1 - concrete[::-1].index("full_attention")
            shared_sliding = len(concrete) - 1 - concrete[::-1].index("sliding_attention")
            for i in range(self.first_kv_shared_layer_idx, len(layer_types)):
                self.layer_idx_to_cache_idx.append(
                    shared_full if layer_types[i] == "full_attention" else shared_sliding
                )

    def _get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        mask = input_ids < self.config.vocab_size_per_layer_input
        tokens = mx.where(mask, input_ids, mx.zeros_like(input_ids))
        out = self.embed_tokens_per_layer(tokens) * (self.config.hidden_size_per_layer_input**0.5)
        return out.reshape(
            *input_ids.shape,
            self.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )

    def _project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: mx.array,
    ) -> mx.array:
        proj = self.per_layer_model_projection(inputs_embeds) * (self.config.hidden_size**-0.5)
        proj = proj.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        return (self.per_layer_projection_norm(proj) + per_layer_inputs) * (2.0**-0.5)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache | None] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) * (self.hidden_size**0.5)
        per_layer_inputs = self._project_per_layer_inputs(h, self._get_per_layer_inputs(inputs))
        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(
            h,
            cache[self.first_full_idx],
            return_array=True,
        )
        sliding_mask = create_attention_mask(
            h,
            cache[self.first_sliding_idx],
            window_size=self.sliding_window,
            return_array=True,
        )
        h0 = h
        target_mag = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5
        h_list = [h0] + [proj(h0) for proj in self.altup_projections]
        h = mx.stack(h_list, axis=0)
        mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
        h = mx.concatenate(
            [
                h[:1],
                h[1:] * (target_mag / mx.maximum(mags, mx.finfo(h0.dtype).min)),
            ],
            axis=0,
        )
        for i, layer in enumerate(self.layers):
            is_global = (self.config.layer_types or [])[i] == "full_attention"
            mask = global_mask if is_global else sliding_mask
            h = layer(
                h,
                mask,
                cache[self.layer_idx_to_cache_idx[i]],
                per_layer_inputs[:, :, i, :],
            )
        target_mag = mx.mean(h[0] ** 2, axis=-1, keepdims=True) ** 0.5
        unembed = [proj(h[i + 1]) for i, proj in enumerate(self.altup_unembed_projections)]
        if unembed:
            unembed_stacked = mx.stack(unembed, axis=0)
            mags = mx.mean(unembed_stacked**2, axis=-1, keepdims=True) ** 0.5
            unembed_stacked = unembed_stacked * (
                target_mag / mx.maximum(mags, mx.finfo(h0.dtype).min)
            )
            h = mx.concatenate([h[:1], unembed_stacked], axis=0)
        h = mx.mean(h, axis=0)
        out = self.norm(h)
        out = self.embed_tokens.as_linear(out)
        if self.config.final_logit_softcapping is not None:
            out = _logit_softcap(self.config.final_logit_softcapping, out)
        return out

    def make_cache(
        self,
    ) -> list[KVCache | RotatingKVCache]:
        layer_types = self.config.layer_types or []
        caches: list[KVCache | RotatingKVCache] = []
        for layer_type in layer_types[: self.first_kv_shared_layer_idx]:
            if layer_type == "full_attention":
                caches.append(KVCache())
            elif layer_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.config.sliding_window, keep=0))
            else:
                raise NotImplementedError(f"Unknown layer type: {layer_type}")
        return caches


class Model(nn.Module):
    """Gemma 3N LM head wrapper — satisfies ModelProtocol (AC17)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        tc = args.text_config or {}
        self._text_config = TextConfig.from_dict(tc)
        self.model = Gemma3nBackbone(self._text_config)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.model(input_ids, cache=cache)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self._text_config.vocab_size

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return self.model.make_cache()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in weights.items() if "vision_tower" not in k}
        out = {k: v for k, v in out.items() if "audio_tower" not in k}
        out = {k: v for k, v in out.items() if "embed_audio" not in k}
        out = {k: v for k, v in out.items() if "embed_vision" not in k}
        return out
