"""MoonViT vision encoder for Kimi-VL.

Ported from mlx-vlm's kimi_vl/vision.py. Features:
- Conv2d patch embedding with learnable 2D interpolated position embeddings
- Complex 2D RoPE
- Block-diagonal attention for multi-image batches
- Spatial patch merger (reshape-based, not MLP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.models.vision.interpolate import bicubic_interpolate


@dataclass
class MoonViTConfig:
    """MoonViT vision encoder configuration."""

    model_type: str = "moonvit"
    depth: int = 27
    embed_dim: int = 1152
    hidden_size: int = 1152
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    mlp_ratio: float = 4.0
    intermediate_size: int = 4304
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    spatial_merge_size: int = 2
    merge_kernel_size: list[int] = field(default_factory=lambda: [2, 2])


def _as_hw_shapes(grid_hws: mx.array) -> list[tuple[int, int]]:
    raw = grid_hws.tolist() if hasattr(grid_hws, "tolist") else grid_hws
    return [(int(s[0]), int(s[1])) for s in raw]


def _make_block_attention_mask(cu_seqlens: mx.array, seq_length: int) -> mx.array:
    mask = mx.zeros((seq_length, seq_length), dtype=mx.bool_)
    for i in range(1, len(cu_seqlens)):
        start = int(cu_seqlens[i - 1])
        end = int(cu_seqlens[i])
        mask[start:end, start:end] = True
    return mask


def _view_as_complex(x: mx.array) -> mx.array:
    return x[..., 0] + 1j * x[..., 1]


def _view_as_real(x: mx.array) -> mx.array:
    return mx.stack([mx.real(x), mx.imag(x)], axis=-1)


def _apply_rope(
    q: mx.array, k: mx.array, freqs_cis: mx.array,
) -> tuple[mx.array, mx.array]:
    freqs_cis = mx.expand_dims(freqs_cis, axis=-2)  # ..., 1, head_dim/2
    q_ = _view_as_complex(q.astype(mx.float32).reshape(*q.shape[:-1], -1, 2))
    k_ = _view_as_complex(k.astype(mx.float32).reshape(*k.shape[:-1], -1, 2))
    q_out = _view_as_real(q_ * freqs_cis).reshape(*q.shape)
    k_out = _view_as_real(k_ * freqs_cis).reshape(*k.shape)
    return q_out.astype(q.dtype), k_out.astype(k.dtype)


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with complex frequency grid."""

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self._freqs_cis: mx.array | None = None
        self._shape_cache: dict[tuple[int, int], mx.array] = {}

    def _precompute_freqs_cis(self) -> mx.array:
        n = self.max_height * self.max_width
        flat_pos = mx.arange(0, n, dtype=mx.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = mx.arange(0, self.dim, 4)[: (self.dim // 4)].astype(mx.float32)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = mx.outer(x_pos, freqs)
        y_freqs = mx.outer(y_pos, freqs)
        x_cis = mx.cos(x_freqs) + 1j * mx.sin(x_freqs)
        y_cis = mx.cos(y_freqs) + 1j * mx.sin(y_freqs)
        freqs_cis = mx.stack([x_cis, y_cis], axis=-1)
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(self, grid_hws: mx.array) -> mx.array:
        if self._freqs_cis is None:
            self._freqs_cis = self._precompute_freqs_cis()
        shapes = _as_hw_shapes(grid_hws)
        parts = []
        for shape in shapes:
            cached = self._shape_cache.get(shape)
            if cached is None:
                h, w = shape
                cached = self._freqs_cis[:h, :w].reshape(-1, self.dim // 2)
                self._shape_cache[shape] = cached
            parts.append(cached)
        return mx.concatenate(parts, axis=0)


class Learnable2DInterpPosEmb(nn.Module):
    """Learnable 2D position embeddings with bicubic interpolation."""

    def __init__(self, height: int, width: int, dim: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.weight = mx.ones((height, width, dim))

    def _get_pos_emb(self, shape: tuple[int, int]) -> mx.array:
        if shape == (self.height, self.width):
            return self.weight.reshape(-1, self.weight.shape[-1])
        # Interpolate: (H, W, D) -> (1, D, H, W)
        w = mx.expand_dims(self.weight.transpose(2, 0, 1), axis=0)
        resized = bicubic_interpolate(w, size=shape)
        return resized.squeeze(0).transpose(1, 2, 0).reshape(-1, self.weight.shape[-1])

    def __call__(self, x: mx.array, grid_hws: mx.array) -> mx.array:
        pos_embs = [self._get_pos_emb(shape) for shape in _as_hw_shapes(grid_hws)]
        return x + mx.concatenate(pos_embs, axis=0).astype(x.dtype)


class PatchEmbed(nn.Module):
    """Conv2d patch embedding with learnable 2D position embedding."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size, bias=True,
        )
        self.pos_emb = Learnable2DInterpPosEmb(
            config.init_pos_emb_height, config.init_pos_emb_height, config.embed_dim,
        )

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states).swapaxes(1, 3)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        hidden_states = self.pos_emb(hidden_states, grid_thw)
        return hidden_states


class MoonViTAttention(nn.Module):
    """Multi-head attention with complex 2D RoPE."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.wqkv = nn.Linear(dim, dim * 3, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        cu_seqlens: mx.array | None = None,
        rotary_pos_emb: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = (
            self.wqkv(x)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        if rotary_pos_emb is not None:
            q, k = _apply_rope(q, k, rotary_pos_emb)

        if attention_mask is None and cu_seqlens is not None:
            attention_mask = _make_block_attention_mask(cu_seqlens, seq_length)

        q = q.transpose(1, 0, 2)[None, ...]
        k = k.transpose(1, 0, 2)[None, ...]
        v = v.transpose(1, 0, 2)[None, ...]

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.wo(output)


class MoonViTMLP(nn.Module):
    """GELU MLP for MoonViT."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc0 = nn.Linear(dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc1(nn.gelu(self.fc0(x)))


class MoonViTBlock(nn.Module):
    """Transformer block with LayerNorm and block attention."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.norm0 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = MoonViTAttention(config.embed_dim, config.num_heads)
        self.mlp = MoonViTMLP(config.embed_dim, config.intermediate_size)

    def __call__(
        self, x: mx.array, cu_seqlens: mx.array,
        rotary_pos_emb: mx.array, attention_mask: mx.array,
    ) -> mx.array:
        x = x + self.attn(
            self.norm0(x), cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask,
        )
        x = x + self.mlp(self.norm1(x))
        return x


def _patch_merger(
    x: mx.array,
    grid_hws: list[tuple[int, int]],
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[mx.array]:
    """Reshape-based spatial patch merger (no learned params)."""
    d_model = x.shape[-1]
    kh, kw = merge_kernel_size

    lengths = [h * w for h, w in grid_hws]
    split_points = []
    running = 0
    for length in lengths[:-1]:
        running += length
        split_points.append(running)

    sequences = mx.split(x, split_points, axis=0) if split_points else [x]
    outputs = []
    for seq, (height, width) in zip(sequences, grid_hws):
        new_h, new_w = height // kh, width // kw
        reshaped = seq.reshape(new_h, kh, new_w, kw, d_model)
        reshaped = mx.transpose(reshaped, (0, 2, 1, 3, 4))
        merged = reshaped.reshape(new_h * new_w, kh * kw, -1)
        outputs.append(merged)
    return outputs


class MoonViTModel(nn.Module):
    """MoonViT vision encoder for Kimi-VL.

    Input: ``pixel_values (N_patches, C, patch_H, patch_W)``, ``grid_thw (N_images, 2|3)``
    Output: list of ``(merged_patches, kernel_area, D)`` per image
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.merge_kernel_size = tuple(config.merge_kernel_size)

        self.patch_embed = PatchEmbed(config)

        head_dim = config.embed_dim // config.num_heads
        self.rope_pos_emb = Rope2DPosEmb(head_dim, 512, 512)

        self.blocks = [MoonViTBlock(config) for _ in range(config.depth)]
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def __call__(
        self, hidden_states: mx.array, grid_thw: mx.array,
    ) -> list[mx.array]:
        hidden_states = self.patch_embed(hidden_states, grid_thw)
        rotary_pos_emb = self.rope_pos_emb.get_freqs_cis(grid_thw)

        # Compute cu_seqlens
        lengths = mx.concatenate((
            mx.zeros((1,), dtype=grid_thw.dtype),
            grid_thw[:, 0] * grid_thw[:, 1],
        ))
        cu_seqlens = mx.cumsum(lengths.astype(mx.int32), axis=0)
        attention_mask = _make_block_attention_mask(cu_seqlens, hidden_states.shape[0])

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens, rotary_pos_emb, attention_mask)

        hidden_states = self.final_layernorm(hidden_states)

        shapes = _as_hw_shapes(grid_thw)
        return _patch_merger(hidden_states, shapes, self.merge_kernel_size)

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            if "patch_embed.proj.weight" in k and v.ndim == 4:
                out_c, kh, kw, _ = v.shape
                if out_c >= kh and out_c >= kw and kh == kw:
                    sanitized[k] = v
                else:
                    sanitized[k] = v.transpose(0, 2, 3, 1)
            elif "vision_tower.blocks" in k:
                if "attn" not in k and ("wqkv" in k or "wo" in k):
                    sanitized[k.replace("wqkv", "attn.wqkv").replace("wo", "attn.wo")] = v
                else:
                    sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized
