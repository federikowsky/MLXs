"""Qwen3 family model — text-only and multimodal config-driven."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.cache.attention_mask import create_attention_mask
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
from mlxs.models.vision.siglip_builder import build_siglip_vision_tower

_VISION_PREFIXES = (
    "visual.",
    "vision_tower.",
    "vision_model.",
    "multi_modal_projector.",
    "mm_projector.",
)
_VISION_EXACT = ("visual", "vision_tower", "vision_model")


@dataclass
class ModelArgs(MultimodalArgsMixin[dict[str, Any]]):
    """Qwen3 family config: flat text-only or nested multimodal config."""

    model_type: str = "qwen3"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 32
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    head_dim: int = 128
    tie_word_embeddings: bool = False
    rope_scaling: dict[str, float | str] | None = None

    image_token_id: int | None = 151655
    video_token_id: int | None = 151656

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        return cls.from_flat_or_nested(
            params,
            default_model_type=params.get("model_type", "qwen3"),
        )

    def resolved_text_args(self) -> ModelArgs:
        """Resolve the language-side config for nested multimodal layouts."""

        if self.text_config:
            return type(self).from_dict(self.text_config)
        return self


class Attention(nn.Module):
    """Multi-head attention with GQA, QK-norm, and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            head_dim,
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

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3Model(nn.Module):
    """Qwen3 transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)

        return self.norm(h)


def _build_language_components(
    args: ModelArgs,
) -> tuple[Qwen3Model, nn.Linear | None]:
    model = Qwen3Model(args)
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
    sanitized = dict(weights)
    if tie_word_embeddings:
        sanitized.pop("lm_head.weight", None)
    return sanitized


class Model(nn.Module):
    """Qwen3 family wrapper — text-only or multimodal via config + model_mode."""

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
        cache: list[KVCache] | None = None,
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
                encoder_kwargs={"grid_thw": image_grid_thw},
            )

        return prepare_multimodal_inputs(
            input_ids,
            embed_tokens=self.model.embed_tokens,
            image_branch=image_branch,
        )

    def _encode_image(
        self,
        pixel_values: mx.array,
        *,
        grid_thw: mx.array | None = None,
    ) -> mx.array:
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        image = pixel_values.astype(dtype)
        grid = grid_thw if grid_thw is not None else mx.array([[1, 1, 1]])
        return self.vision_tower(image, grid)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

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
            elif key.startswith("language_model."):
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
    def layers(self) -> list[TransformerBlock]:
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
