"""LFM2-VL SigLIP2-style vision encoder.

Ported from mlx-vlm's lfm2_vl/vision.py. Standard ViT with:
- Linear patch embedding (not Conv2d)
- Learnable position embeddings with bicubic interpolation
- GELU activation MLP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.models.vision.interpolate import bicubic_interpolate


@dataclass
class LFM2VisionConfig:
    """LFM2-VL vision encoder configuration."""

    model_type: str = "lfm2_vl"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 256
    layer_norm_eps: float = 1e-6


class LFM2Attention(nn.Module):
    """Multi-head self-attention for LFM2 ViT."""

    def __init__(self, config: LFM2VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // self.num_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class LFM2MLP(nn.Module):
    """GELU MLP for LFM2 ViT."""

    def __init__(self, config: LFM2VisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class LFM2EncoderLayer(nn.Module):
    """Single transformer block in LFM2 ViT."""

    def __init__(self, config: LFM2VisionConfig) -> None:
        super().__init__()
        self.self_attn = LFM2Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = LFM2MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class LFM2VisionEmbeddings(nn.Module):
    """Patch embedding with learnable position embeddings and bicubic interpolation."""

    def __init__(self, config: LFM2VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.position_embedding_size = int(config.num_patches**0.5)

        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size * config.patch_size,
            config.hidden_size,
        )
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)

    def __call__(
        self, pixel_values: mx.array, spatial_shapes: mx.array | None = None,
    ) -> mx.array:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.astype(target_dtype))

        pos_emb = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1,
        )

        if spatial_shapes is not None:
            batch_size = spatial_shapes.shape[0]
            embed_dim = pos_emb.shape[-1]
            max_length = pixel_values.shape[1]

            # (H, W, D) -> (1, D, H, W)
            pos_4d = mx.expand_dims(pos_emb.transpose(2, 0, 1), axis=0)

            resized_list = []
            for i in range(batch_size):
                h, w = int(spatial_shapes[i][0]), int(spatial_shapes[i][1])
                resized = bicubic_interpolate(pos_4d, size=(h, w))
                flat = resized.reshape(embed_dim, h * w).transpose(1, 0)  # (h*w, D)
                # Pad to max_length
                if flat.shape[0] < max_length:
                    pad = mx.broadcast_to(flat[0:1], (max_length - flat.shape[0], embed_dim))
                    flat = mx.concatenate([flat, pad], axis=0)
                resized_list.append(flat[:max_length])

            resized_pos = mx.stack(resized_list, axis=0)
            return patch_embeds + resized_pos
        else:
            # Use position embeddings directly
            pos = self.position_embedding.weight[:pixel_values.shape[1]]
            return patch_embeds + pos[None, :]


class LFM2VisionModel(nn.Module):
    """LFM2-VL vision encoder.

    Input: ``pixel_values (B, N_patches, C*P*P)``
    Output: ``(encoder_states, embeddings, last_hidden_state)``
    """

    def __init__(self, config: LFM2VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = LFM2VisionEmbeddings(config)
        self.layers = [LFM2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        pixel_values: mx.array,
        spatial_shapes: mx.array | None = None,
        output_hidden_states: bool = True,
    ) -> tuple[tuple[mx.array, ...], mx.array, mx.array]:
        x = self.embeddings(pixel_values, spatial_shapes=spatial_shapes)
        x = x.astype(self.embeddings.patch_embedding.weight.dtype)

        encoder_states: tuple[mx.array, ...] = (x,) if output_hidden_states else ()
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        last_hidden = self.post_layernorm(x)
        return encoder_states, x, last_hidden

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in weights.items() if "position_ids" not in k}
