"""Pixtral vision encoder — ViT with 2D RoPE and variable-resolution support.

Ported from mlx-vlm's pixtral/vision.py. Used by: pixtral, mistral3.
Supports block-diagonal attention for multi-image batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class PixtralVisionConfig:
    """Pixtral vision encoder configuration."""

    model_type: str = "pixtral"
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    head_dim: int = 64
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    projection_dim: int = 768
    num_channels: int = 3
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


def _position_ids_in_meshgrid(
    patch_embeds_list: list[mx.array], max_width: int,
) -> mx.array:
    """Compute 2D position IDs for variable-resolution patch grids."""
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[0], patch.shape[1]
        h_grid, v_grid = mx.meshgrid(
            mx.arange(height), mx.arange(width), indexing="ij",
        )
        ids = (h_grid.reshape(-1, 1) * max_width + v_grid.reshape(-1, 1)).flatten()
        positions.append(ids)
    return mx.concatenate(positions)


def _generate_block_attention_mask(
    patch_counts: list[int], tensor: mx.array,
) -> mx.array:
    """Block-diagonal mask preventing cross-image attention."""
    seq_len = tensor.shape[1]
    d_min = -1e9

    causal_mask = mx.full((seq_len, seq_len), vals=d_min)

    block_end_idx = mx.cumsum(mx.array(patch_counts))
    block_start_idx = mx.concatenate(
        [mx.array([0]), mx.array(patch_counts[:-1])],
    )
    block_start_idx = mx.cumsum(block_start_idx)

    for start, end in zip(block_start_idx, block_end_idx):
        start, end = int(start), int(end)
        causal_mask[start:end, start:end] = 0

    return mx.broadcast_to(
        causal_mask[None, None, :, :],
        (tensor.shape[0], 1, seq_len, seq_len),
    ).astype(tensor.dtype)


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array,
) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=0)  # unsqueeze head dim
    sin = mx.expand_dims(sin, axis=0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class PixtralRotaryEmbedding:
    """2D rotary position embedding for Pixtral ViT."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        dim = config.head_dim
        base = config.rope_theta
        max_patches_per_side = config.image_size // config.patch_size

        freqs = 1.0 / (
            base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)
        )

        h = mx.arange(max_patches_per_side)
        w = mx.arange(max_patches_per_side)

        freqs_h = mx.outer(h, freqs[::2]).astype(mx.float32)
        freqs_w = mx.outer(w, freqs[1::2]).astype(mx.float32)
        inv_freq = mx.concatenate(
            [
                mx.tile(freqs_h[:, None, :], (1, max_patches_per_side, 1)),
                mx.tile(freqs_w[None, :, :], (max_patches_per_side, 1, 1)),
            ],
            axis=-1,
        ).reshape(-1, dim // 2)

        self.inv_freq = mx.concatenate((inv_freq, inv_freq), axis=-1)

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        freqs = self.inv_freq[position_ids]
        cos = mx.cos(freqs).astype(x.dtype)
        sin = mx.sin(freqs).astype(x.dtype)
        return cos, sin


class PixtralAttention(nn.Module):
    """Multi-head attention with 2D RoPE for Pixtral ViT."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
    ) -> mx.array:
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        B, L, _ = queries.shape
        _, S, _ = keys.shape

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        queries, keys = _apply_rotary_pos_emb(queries, keys, cos, sin)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class PixtralMLP(nn.Module):
    """SwiGLU MLP for Pixtral ViT."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        hidden_dim = config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class PixtralEncoderLayer(nn.Module):
    """Single transformer block in Pixtral ViT."""

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        self.attention = PixtralAttention(config)
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = PixtralMLP(config)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
    ) -> mx.array:
        y = self.attention(
            self.attention_norm(x), self.attention_norm(x), self.attention_norm(x),
            position_embeddings, mask,
        )
        x = x + y
        y = self.feed_forward(self.ffn_norm(x))
        return x + y


class PixtralVisionModel(nn.Module):
    """Pixtral vision encoder with variable-resolution and 2D RoPE.

    Input: ``pixel_values (N, C, H, W)``, ``image_sizes list[(H, W)]``
    Output: ``(1, total_patches, hidden_size)``
    """

    def __init__(self, config: PixtralVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = nn.RMSNorm(config.hidden_size)
        self.layers = [PixtralEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

    def __call__(
        self,
        pixel_values: mx.array,
        image_sizes: list[tuple[int, int]] | mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, ...] | None]:
        if pixel_values.dtype != self.patch_conv.weight.dtype:
            pixel_values = pixel_values.astype(self.patch_conv.weight.dtype)

        # pixel_values: (N, C, H, W) -> transpose to (N, H, W, C) for Conv2d
        x = pixel_values.transpose(0, 2, 3, 1)

        if image_sizes is None:
            image_sizes = [(pixel_values.shape[2], pixel_values.shape[3])] * pixel_values.shape[0]
        else:
            normalized = []
            for s in image_sizes:
                s_val = s.tolist() if hasattr(s, "tolist") else s
                normalized.append((int(s_val[0]), int(s_val[1])))
            image_sizes = normalized

        patch_embeds = self.patch_conv(x)

        # Crop each image's patches to actual resolution
        patch_embeds_list = [
            patch_embeds[i][
                : (image_sizes[i][0] // self.patch_size),
                : (image_sizes[i][1] // self.patch_size),
            ]
            for i in range(len(image_sizes))
        ]

        # Flatten and concatenate all patches
        patch_embeds_flat = mx.concatenate(
            [p.reshape(-1, p.shape[-1]) for p in patch_embeds_list], axis=0,
        )[None, ...]

        patch_embeds_flat = self.ln_pre(patch_embeds_flat)

        # 2D position IDs
        max_width = self.config.image_size // self.config.patch_size
        position_ids = _position_ids_in_meshgrid(patch_embeds_list, max_width)
        position_embedding = self.patch_positional_embedding(patch_embeds_flat, position_ids)

        # Block attention mask
        patch_counts = [p.shape[0] * p.shape[1] for p in patch_embeds_list]
        mask = _generate_block_attention_mask(patch_counts, patch_embeds_flat)

        encoder_states = (patch_embeds_flat,) if output_hidden_states else None

        h = patch_embeds_flat
        for layer in self.layers:
            h = layer(h, position_embeddings=position_embedding, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (h,)  # type: ignore[operator]

        return h, encoder_states

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Sanitize vision tower weights (conv2d transpose, position_ids removal)."""
        sanitized: dict[str, Any] = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            if "patch_conv.weight" in k and v.ndim == 4:
                # PyTorch: (out, in, kH, kW) -> MLX: (out, kH, kW, in)
                out_c, kh, kw, _ = v.shape
                if out_c >= kh and out_c >= kw and kh == kw:
                    sanitized[k] = v
                else:
                    sanitized[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized[k] = v
        return sanitized
