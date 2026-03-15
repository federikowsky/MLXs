"""Qwen3.5 family model — text-only and multimodal config-driven pilot.

Alternating GatedDeltaNet (linear) and full attention every
``full_attention_interval``. In multimodal mode, the same language backbone is
paired with a SigLIP vision tower for image/video placeholder merging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.gated_delta import gated_delta_update
from mlxs.layers.rope import initialize_rope
from mlxs.models.multimodal_shared import (
    MediaBranch,
    MultimodalArgsMixin,
    build_dual_mode_components,
    prepare_multimodal_inputs,
)
from mlxs.models.vision.siglip_builder import build_siglip_vision_tower

_VISION_PREFIXES = (
    "visual.",
    "vision_tower.",
    "vision_model.",
    "multi_modal_projector.",
    "mm_projector.",
)
_VISION_EXACT = ("visual", "vision_tower", "vision_model")

# ----- Model args -----


@dataclass
class ModelArgs(MultimodalArgsMixin[dict[str, Any]]):
    """Qwen3.5 family config: flat text-only or nested multimodal config."""

    model_type: str = "qwen3_5"
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: int | None = None
    full_attention_interval: int = 4
    rope_theta: float = 100000.0
    partial_rotary_factor: float = 0.25
    rope_scaling: dict[str, Any] | None = None
    rope_parameters: dict[str, Any] | None = None
    image_token_id: int | None = 248056
    video_token_id: int | None = 248057

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        return cls.from_flat_or_nested(
            params,
            default_model_type=params.get("model_type", "qwen3_5"),
            vision_config_keys=("vision_config", "visual_config"),
        )

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters is not None:
            params = self.rope_parameters
            if "type" not in params and "rope_type" in params:
                params["type"] = params.pop("rope_type")
            self.partial_rotary_factor = params.get("partial_rotary_factor", 0.25)
            self.rope_theta = params.get("rope_theta", 100000.0)
            self.rope_scaling = params

    def resolved_text_args(self) -> ModelArgs:
        """Resolve the language-side config for nested multimodal layouts."""

        if self.text_config:
            return type(self).from_dict(self.text_config)
        return self


# ----- RMSNormGated (model-specific, not in layers/norms) -----


class RMSNormGated(nn.Module):
    """RMSNorm then optional gate: out = sigmoid(gate) * rms_norm(x) or x."""

    def __init__(self, dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array, gate: mx.array | None = None) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return (nn.silu(gate.astype(mx.float32)) * x.astype(mx.float32)).astype(
                hidden_states.dtype
            )
        return x.astype(hidden_states.dtype)


# ----- GatedDeltaNet (linear layer) -----


class GatedDeltaNet(nn.Module):
    """Linear attention block: conv on qkv, then gated delta recurrence."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"linear_num_value_heads ({self.num_v_heads}) must be divisible "
                f"by linear_num_key_heads ({self.num_k_heads})"
            )
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,))
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)
        self.norm = RMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]
        state = cache[1] if cache is not None else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask)
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


# ----- Attention (full attention layer) -----


class Attention(nn.Module):
    """Full attention with QK-norm, partial RoPE, and output gate (sigmoid)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        rope_dims = int(self.head_dim * args.partial_rotary_factor)
        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            rope_dims,
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
        q_proj_out = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_out.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
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
        return self.o_proj(output * mx.sigmoid(gate))


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


# ----- Decoder layer -----


class DecoderLayer(nn.Module):
    """One block: either GatedDeltaNet or Attention, then MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


# ----- Backbone and Model -----


class Qwen35Model(nn.Module):
    """Qwen3.5 transformer backbone: embed -> alternating linear/attn layers -> norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._ssm_idx = 0
        self._fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        fa_mask = create_attention_mask(h, cache[self._fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self._ssm_idx])
        for layer, c in zip(self.layers, cache, strict=True):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


def _build_language_components(
    args: ModelArgs,
) -> tuple[Qwen35Model, nn.Linear | None]:
    model = Qwen35Model(args)
    lm_head = (
        None
        if args.tie_word_embeddings
        else nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    )
    return model, lm_head


def _sanitize_text_weights(
    weights: dict[str, Any],
    *,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    has_mtp_weights = any("mtp." in key for key in weights)
    has_unsanitized_conv1d = any(
        "conv1d.weight" in key and value.shape[-1] != 1
        for key, value in weights.items()
    )
    should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
    sanitized = {key: value for key, value in weights.items() if "mtp." not in key}
    if tie_word_embeddings:
        sanitized.pop("lm_head.weight", None)
        sanitized.pop("language_model.lm_head.weight", None)

    norm_keys = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for key, value in list(sanitized.items()):
        if "conv1d.weight" in key and value.shape[-1] != 1:
            sanitized[key] = value.moveaxis(2, 1)
        if (
            should_shift_norm_weights
            and any(key.endswith(suffix) for suffix in norm_keys)
            and value.ndim == 1
        ):
            sanitized[key] = value + 1.0
    return sanitized


class Model(nn.Module):
    """Qwen3.5 family wrapper — text-only or multimodal via config + model_mode."""

    def __init__(
        self,
        args: ModelArgs,
        *,
        model_mode: ModelMode = ModelMode.TEXT,
    ) -> None:
        super().__init__()
        self.config = args
        self.model_type = args.model_type
        self._image_token_id = args.image_token_id
        self._video_token_id = args.video_token_id

        text_args = args.resolved_text_args()
        self.args = text_args

        components = build_dual_mode_components(
            model_mode=model_mode,
            language_builder=lambda: _build_language_components(text_args),
            vision_config=args.vision_config,
            vision_builder=build_siglip_vision_tower,
        )
        self._mode = components.model_mode
        self.model, self.lm_head = components.language_model
        if components.vision_tower is not None:
            self.vision_tower = components.vision_tower

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        if self.lm_head is None:
            raise ValueError("lm_head is required when tie_word_embeddings is False")
        return self.lm_head(out)

    def prepare_inputs(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        video_pixel_values: mx.array | None = None,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array | None]:
        if not self.supports_vision or (pixel_values is None and video_pixel_values is None):
            return input_ids, None

        image_branch = None
        if pixel_values is not None and self._image_token_id is not None:
            image_branch = MediaBranch(
                values=pixel_values,
                placeholder_token_id=self._image_token_id,
                encode=self._encode_media,
                encoder_kwargs={"grid_thw": image_grid_thw},
            )

        video_branch = None
        if video_pixel_values is not None and self._video_token_id is not None:
            video_branch = MediaBranch(
                values=video_pixel_values,
                placeholder_token_id=self._video_token_id,
                encode=self._encode_media,
                encoder_kwargs={"grid_thw": kwargs.get("video_grid_thw", image_grid_thw)},
            )

        return prepare_multimodal_inputs(
            input_ids,
            embed_tokens=self.model.embed_tokens,
            image_branch=image_branch,
            video_branch=video_branch,
        )

    def _encode_media(
        self,
        pixel_values: mx.array,
        *,
        grid_thw: mx.array | None = None,
    ) -> mx.array:
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        media = pixel_values.astype(dtype)
        grid = grid_thw if grid_thw is not None else mx.array([[1, 1, 1]])
        return self.vision_tower(media, grid)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return [
            ArraysCache(size=2) if layer.is_linear else KVCache() for layer in self.model.layers
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if self._mode == ModelMode.TEXT:
            filtered = {
                key: value
                for key, value in weights.items()
                if key not in _VISION_EXACT
                and not any(key.startswith(prefix) for prefix in _VISION_PREFIXES)
            }
            if self.model_type in {"qwen3_5", "qwen3_5_vl"}:
                filtered = {
                    (
                        key[len("language_model.") :]
                        if key.startswith("language_model.")
                        else key
                    ): value
                    for key, value in filtered.items()
                }
            return _sanitize_text_weights(
                filtered,
                tie_word_embeddings=self.args.tie_word_embeddings,
            )

        language_weights: dict[str, Any] = {}
        vision_weights: dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith(("multi_modal_projector.", "mm_projector.")):
                continue
            if key.startswith("visual."):
                key = f"vision_tower.{key[len('visual.') :]}"
            elif key.startswith("vision_model."):
                key = f"vision_tower.{key[len('vision_model.') :]}"

            if key.startswith("vision_tower."):
                vision_weights[key] = value
            elif (
                self.model_type in {"qwen3_5", "qwen3_5_vl"}
                and key.startswith("language_model.")
            ):
                language_weights[key[len("language_model.") :]] = value
            else:
                language_weights[key] = value

        sanitized_language = _sanitize_text_weights(
            language_weights,
            tie_word_embeddings=self.args.tie_word_embeddings,
        )
        if hasattr(self, "vision_tower"):
            vision_weights = self.vision_tower.sanitize(vision_weights)
        return sanitized_language | vision_weights

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    @property
    def supports_vision(self) -> bool:
        return self._mode != ModelMode.TEXT and hasattr(self, "vision_tower")

    @property
    def supports_audio(self) -> bool:
        return False

    @property
    def image_token_id(self) -> int | None:
        return self._image_token_id if self.supports_vision else None

    @property
    def language_model(self) -> Model:
        """Compatibility alias for legacy wrapper-style access patterns."""

        return self


__all__ = [
    "MLP",
    "Attention",
    "DecoderLayer",
    "GatedDeltaNet",
    "Model",
    "ModelArgs",
    "Qwen35Model",
    "RMSNormGated",
]
