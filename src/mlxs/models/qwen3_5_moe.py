"""Qwen3.5 MoE family model: text-only or multimodal via config + model_mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.family_adapters import sanitize_qwen35_moe_family_weights
from mlxs.layers.moe import SwitchGLU
from mlxs.models.multimodal_shared import (
    MediaBranch,
    MultimodalArgsMixin,
    build_dual_mode_components,
    prepare_multimodal_inputs,
)
from mlxs.models.qwen3_5 import (
    MLP,
    Attention,
    GatedDeltaNet,
)
from mlxs.models.qwen3_5 import (
    Model as Qwen35LanguageModel,
)
from mlxs.models.qwen3_5 import (
    ModelArgs as Qwen35ModelArgs,
)
from mlxs.models.vision.siglip_builder import build_siglip_vision_tower


@dataclass
class TextModelArgs(Qwen35ModelArgs):
    """Qwen3.5-MoE text config from ``text_config``."""

    model_type: str = "qwen3_5_moe"
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    mlp_only_layers: list[int] = field(default_factory=list)
    rope_parameters: dict[str, Any] | None = field(
        default_factory=lambda: {
            "type": "default",
            "rope_theta": 100000.0,
            "partial_rotary_factor": 0.25,
        }
    )


class SparseMoeBlock(nn.Module):
    """Sparse MoE with top-k routing, SwitchGLU experts, and one shared expert."""

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        shared_size = args.shared_expert_intermediate_size or args.intermediate_size

        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, self.num_experts)
        self.shared_expert = MLP(dim, shared_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = mx.softmax(self.gate(x), axis=-1)
        inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        routed = self.switch_mlp(x, inds)
        routed = (routed * scores[..., None]).sum(axis=-2)

        shared = self.shared_expert(x)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
        return routed + shared


class DecoderLayer(nn.Module):
    """Hybrid decoder layer with dense or sparse feed-forward block."""

    def __init__(self, args: TextModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        is_moe = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.mlp: SparseMoeBlock | MLP = (
            SparseMoeBlock(args) if is_moe else MLP(args.hidden_size, args.intermediate_size)
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        if self.is_linear:
            residual = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            residual = self.self_attn(self.input_layernorm(x), mask, cache)
        hidden = x + residual
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TextModel(nn.Module):
    """Qwen3.5-MoE text backbone."""

    def __init__(self, args: TextModelArgs) -> None:
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
        hidden = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        fa_mask = create_attention_mask(hidden, cache[self._fa_idx])
        ssm_mask = cast(mx.array | None, create_ssm_mask(hidden, cache[self._ssm_idx]))
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            if layer.is_linear:
                hidden = layer(
                    hidden,
                    mask=ssm_mask,
                    cache=cast(ArraysCache | None, layer_cache),
                )
            else:
                hidden = layer(
                    hidden,
                    mask=fa_mask,
                    cache=cast(KVCache | None, layer_cache),
                )
        return self.norm(hidden)


@dataclass
class ModelArgs(MultimodalArgsMixin[dict[str, Any]]):
    """Top-level Qwen3.5-MoE config with optional multimodal envelope."""

    model_type: str = "qwen3_5_moe"
    text_config: dict[str, Any] = field(default_factory=dict)
    image_token_id: int | None = 248056
    video_token_id: int | None = 248057

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        return cls.from_flat_or_nested(
            params,
            default_model_type=params.get("model_type", "qwen3_5_moe"),
            vision_config_keys=("vision_config", "visual_config"),
        )

    def resolved_text_args(self) -> TextModelArgs:
        if self.text_config:
            return cast(TextModelArgs, TextModelArgs.from_dict(self.text_config))
        return cast(TextModelArgs, TextModelArgs.from_dict({"model_type": self.model_type}))


def _build_language_model(args: TextModelArgs) -> Qwen35LanguageModel:
    language_model = Qwen35LanguageModel(args)
    language_model.model = TextModel(args)
    language_model.args = args
    return language_model


class Model(nn.Module):
    """Qwen3.5-MoE wrapper with optional SigLIP image/video support."""

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
            language_builder=lambda: _build_language_model(text_args),
            vision_config=args.vision_config,
            vision_builder=build_siglip_vision_tower,
        )
        self._mode = components.model_mode
        self.language_model = components.language_model
        if components.vision_tower is not None:
            self.vision_tower = components.vision_tower

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
        )

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
            embed_tokens=self.language_model.model.embed_tokens,
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
        return self.language_model.num_layers

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return self.language_model.make_cache()

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.language_model.layers

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return sanitize_qwen35_moe_family_weights(
            weights,
            model_mode=self._mode,
            tie_word_embeddings=self.args.tie_word_embeddings,
            vision_sanitize=(
                self.vision_tower.sanitize
                if hasattr(self, "vision_tower")
                else None
            ),
        )

    @property
    def supports_vision(self) -> bool:
        return self._mode != ModelMode.TEXT and hasattr(self, "vision_tower")

    @property
    def supports_audio(self) -> bool:
        return False

    @property
    def image_token_id(self) -> int | None:
        return self._image_token_id if self.supports_vision else None
