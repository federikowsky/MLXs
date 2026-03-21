"""LFM2 family model — text-only and multimodal config-driven."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.rope import initialize_rope
from mlxs.models.multimodal_shared import (
    MediaBranch,
    MultimodalArgsMixin,
    build_dual_mode_components,
    prepare_multimodal_inputs,
)

if TYPE_CHECKING:
    from mlxs.models.vision.lfm2_vit import LFM2VisionModel
    from mlxs.models.vision.projectors import MLPProjector

_VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")
_VISION_EXACT = ("vision_tower", "multi_modal_projector")


@dataclass
class ModelArgs(MultimodalArgsMixin[dict[str, Any]]):
    """LFM2 config; supports flat text-only and nested multimodal layouts."""

    model_type: str = "lfm2"
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 131072
    norm_eps: float = 1e-6
    conv_bias: bool = False
    conv_L_cache: int = 4
    block_dim: int = 2048
    block_ff_dim: int = 8192
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float | None = None
    block_auto_adjust_ff_dim: bool = True
    rope_theta: float = 1000000.0
    rope_parameters: dict[str, Any] | None = None
    full_attn_idxs: list[int] | None = None
    layer_types: list[str] | None = None
    image_token_id: int | None = 151655
    vision_feature_layer: int = -1

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        return cls.from_flat_or_nested(
            params,
            default_model_type=params.get("model_type", "lfm2"),
        )

    def __post_init__(self) -> None:
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = float(self.rope_parameters["rope_theta"])
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is not None and self.full_attn_idxs is None:
            self.full_attn_idxs = [
                i for i, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            ]
        if self.full_attn_idxs is None:
            self.full_attn_idxs = []

    def resolved_text_args(self) -> ModelArgs:
        """Resolve the language-side config for nested multimodal layouts."""

        if self.text_config:
            return type(self).from_dict(self.text_config)
        return self


class Attention(nn.Module):
    """Multi-head attention with per-head Q/K RMSNorm and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_layernorm(
            queries.reshape(batch_size, seq_len, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_layernorm(
            keys.reshape(batch_size, seq_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(queries, keys, values, cache, self.scale, mask)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    """Gated depthwise conv block with recurrent state in ArraysCache."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.conv = nn.Conv1d(
            args.hidden_size,
            args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=args.conv_bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=args.conv_bias)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        gate_1, gate_2, inputs = mx.split(self.in_proj(x), 3, axis=-1)
        gated = gate_1 * inputs
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0.0)

        if cache is not None:
            state = cache[0]
            if state is None:
                state = mx.zeros(
                    (batch_size, self.L_cache - 1, self.args.hidden_size),
                    dtype=gated.dtype,
                )
            gated = mx.concatenate([state, gated], axis=1)
            n_keep = self.L_cache - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, seq_len)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(gated, positions, axis=1)
            else:
                cache[0] = gated[:, -n_keep:, :]
            cache.advance(seq_len)
        else:
            gated = mx.pad(gated, [(0, 0), (self.L_cache - 1, 0), (0, 0)])

        conv_out = self.conv(gated)
        return self.out_proj(gate_2 * conv_out)


class MLP(nn.Module):
    """SwiGLU MLP with optional auto-adjusted intermediate size."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: float | None,
    ) -> None:
        super().__init__()
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class Lfm2DecoderLayer(nn.Module):
    """Single decoder layer: either Attention or ShortConv, then MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)

        self.feed_forward = MLP(
            dim=args.block_dim,
            ff_dim=args.block_ff_dim,
            multiple_of=args.block_multiple_of,
            auto_adjust_ff_dim=args.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=args.block_ffn_dim_multiplier,
        )
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        if self.is_attention_layer:
            residual = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            residual = self.conv(self.operator_norm(x), mask=mask, cache=cache)
        hidden = x + residual
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class Lfm2Model(nn.Module):
    """LFM2 transformer backbone (embed + alternating conv/attn layers)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(args, layer_idx=index)
            for index in range(args.num_hidden_layers)
        ]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self._fa_idx = args.full_attn_idxs[0] if args.full_attn_idxs else 0
        self._conv_idx = next(
            (index for index in range(args.num_hidden_layers) if index not in args.full_attn_idxs),
            0,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        attn_mask = create_attention_mask(hidden, cache[self._fa_idx])
        conv_mask = create_ssm_mask(hidden, cache[self._conv_idx])

        for layer, layer_cache in zip(self.layers, cache, strict=True):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)

        return self.embedding_norm(hidden)


@dataclass(frozen=True, slots=True)
class LFM2VisionComponents:
    vision_tower: LFM2VisionModel
    projector: MLPProjector


def _build_vision_components(
    raw_config: dict[str, Any],
    *,
    text_hidden: int,
) -> LFM2VisionComponents:
    from mlxs.models.vision.lfm2_vit import LFM2VisionConfig, LFM2VisionModel
    from mlxs.models.vision.projectors import MLPProjector

    config = LFM2VisionConfig(
        **{
            key: value
            for key, value in raw_config.items()
            if key in LFM2VisionConfig.__dataclass_fields__
        }
    )
    vision_tower = LFM2VisionModel(config)
    projector = MLPProjector(
        in_dim=config.hidden_size,
        hidden_dim=text_hidden,
        out_dim=text_hidden,
    )
    return LFM2VisionComponents(vision_tower=vision_tower, projector=projector)


def _sanitize_text_weights(weights: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for name, param in weights.items():
        if "conv.weight" in name and param.shape[-1] > param.shape[1]:
            param = param.transpose(0, 2, 1)
        sanitized[name] = param
    return sanitized


class Model(nn.Module):
    """LFM2 family wrapper — text-only or multimodal via config + model_mode."""

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

        text_args = args.resolved_text_args()
        self.args = text_args

        components = build_dual_mode_components(
            model_mode=model_mode,
            language_builder=lambda: Lfm2Model(text_args),
            vision_config=args.vision_config,
            vision_builder=lambda raw_config: _build_vision_components(
                raw_config,
                text_hidden=text_args.hidden_size,
            ),
        )
        self._mode = components.model_mode
        self.model = components.language_model
        if components.vision_tower is not None:
            self.vision_tower = components.vision_tower.vision_tower
            self.multi_modal_projector = components.vision_tower.projector

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | ArraysCache] | None = None,
        mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache, input_embeddings=input_embeddings)
        return self.model.embed_tokens.as_linear(out)

    def prepare_inputs(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        **_kwargs: Any,
    ) -> tuple[mx.array, mx.array | None]:
        if not self.supports_vision or pixel_values is None:
            return input_ids, None

        image_branch = None
        if self._image_token_id is not None:
            image_branch = MediaBranch(
                values=pixel_values,
                placeholder_token_id=self._image_token_id,
                encode=self._encode_image,
            )

        return prepare_multimodal_inputs(
            input_ids,
            embed_tokens=self.model.embed_tokens,
            image_branch=image_branch,
        )

    def _encode_image(self, pixel_values: mx.array) -> mx.array:
        vision_inputs, spatial_shapes = self._prepare_vision_inputs(pixel_values)
        _, _, last_hidden = self.vision_tower(vision_inputs, spatial_shapes=spatial_shapes)
        image_features = self.multi_modal_projector(last_hidden)
        return image_features.reshape(-1, image_features.shape[-1])

    def _prepare_vision_inputs(
        self,
        pixel_values: mx.array,
    ) -> tuple[mx.array, mx.array | None]:
        if pixel_values.ndim == 2:
            return pixel_values[None, ...], None
        if pixel_values.ndim == 3:
            return pixel_values, None
        if pixel_values.ndim != 4:
            raise ValueError(
                f"lfm2 expects 2D/3D patch tensors or 4D images, got ndim={pixel_values.ndim}"
            )

        batch_size, channels, height, width = pixel_values.shape
        patch_size = self.vision_tower.config.patch_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "lfm2 pixel_values height/width must be multiples of the vision patch size"
            )

        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = pixel_values.reshape(
            batch_size,
            channels,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(batch_size, grid_h * grid_w, channels * patch_size * patch_size)
        spatial_shapes = mx.array([[grid_h, grid_w]] * batch_size, dtype=mx.int32)
        return patches, spatial_shapes

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return [
            KVCache() if layer.is_attention_layer else ArraysCache(size=1)
            for layer in self.model.layers
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if self._mode == ModelMode.TEXT:
            filtered = {
                key: value
                for key, value in weights.items()
                if key not in _VISION_EXACT
                and not any(key.startswith(prefix) for prefix in _VISION_PREFIXES)
            }
            filtered = {
                (
                    key[len("language_model.") :]
                    if key.startswith("language_model.")
                    else key
                ): value
                for key, value in filtered.items()
            }
            return _sanitize_text_weights(filtered)

        language_weights: dict[str, Any] = {}
        vision_weights: dict[str, Any] = {}
        projector_weights: dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith("vision_tower."):
                vision_weights[key] = value
            elif key.startswith("multi_modal_projector."):
                projector_weights[key] = value
            elif key.startswith("language_model."):
                language_weights[key[len("language_model.") :]] = value
            else:
                language_weights[key] = value

        sanitized_language = _sanitize_text_weights(language_weights)
        if hasattr(self, "vision_tower"):
            vision_weights = self.vision_tower.sanitize(vision_weights)
        return sanitized_language | vision_weights | projector_weights

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
    def layers(self) -> list[Lfm2DecoderLayer]:
        return self.model.layers

    @property
    def language_model(self) -> Model:
        """Compatibility alias for legacy wrapper-style access patterns."""

        return self

    def parameters(self) -> dict[str, Any]:
        return dict(self.trainable_parameters())
