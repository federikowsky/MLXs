"""SigLIP / Qwen2-VL vision encoder — ported from mlx-vlm (§7.4, FR12).

Conv3d patch embedding, 2D rotary position embeddings, block-diagonal
attention, and spatial patch merger. Used by: qwen2_vl, qwen3_vl,
qwen3_vl_moe.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    """Vision encoder configuration (from config.json ``vision_config``)."""

    model_type: str = "qwen2_vl"
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    in_channels: int = 3
    mlp_ratio: float = 4.0
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


# ---------- Helpers ----------


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb_vision(tensor: mx.array, freqs: mx.array) -> mx.array:
    orig_dtype = tensor.dtype
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    # freqs: (seq, head_dim//2) -> expand for heads and tile for complex pairs
    cos = mx.expand_dims(cos, axis=1)   # (seq, 1, head_dim//2)
    cos = mx.tile(cos, (1, 1, 2))       # (seq, 1, head_dim)
    cos = mx.expand_dims(cos, axis=0)   # (1, seq, 1, head_dim)
    sin = mx.expand_dims(sin, axis=1)
    sin = mx.tile(sin, (1, 1, 2))
    sin = mx.expand_dims(sin, axis=0)
    output = (tensor * cos) + (_rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


# ---------- Modules ----------


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for 2D spatial positions."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen if isinstance(seqlen, int) else seqlen.tolist(), dtype=inv_freq.dtype)
        return mx.outer(seq, inv_freq)


class PatchEmbed(nn.Module):
    """3D convolution patch embedding for images/video frames."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=kernel_size, stride=kernel_size, bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1, self.in_channels, self.temporal_patch_size,
            self.patch_size, self.patch_size,
        ).moveaxis(1, 4)
        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(-1, self.embed_dim)


class PatchMerger(nn.Module):
    """Spatial merge: group ``spatial_merge_size^2`` patches and project."""

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionAttention(nn.Module):
    """Multi-head attention with block-diagonal masking via cu_seqlens."""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self, x: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array,
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        q, k, v = mx.split(qkv, 3)

        q = _apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = _apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        # (seq, heads, head_dim) -> (heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Block-diagonal attention per image/video segment
        splits = [
            mx.split(t, cu_seqlens[1:-1].tolist(), axis=2) for t in (q, k, v)
        ]
        attn_outputs = []
        for qi, ki, vi in zip(*splits):
            out = mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=self.scale, mask=None)
            attn_outputs.append(out)
        output = mx.concatenate(attn_outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)


class VisionMLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approx="fast")
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class Qwen2VLVisionBlock(nn.Module):
    """Pre-norm transformer block for the vision encoder."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden = int(config.embed_dim * config.mlp_ratio)
        self.attn = VisionAttention(dim=config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMLP(dim=config.embed_dim, hidden_dim=mlp_hidden)

    def __call__(
        self, hidden_states: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class SigLIPVisionModel(nn.Module):
    """Complete vision encoder: patch embed -> transformer blocks -> merger.

    Args:
        config: VisionConfig from the model's ``vision_config``.

    Input:
        pixel_values: ``(N_patches, C * temporal_patch_size, patch_H, patch_W)``
        grid_thw: ``(N_images, 3)`` — ``[temporal, height, width]`` per image.

    Output:
        ``(N_merged_patches, hidden_size)`` — merged patch features.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
        )

    def _compute_rotary_pos_emb(self, grid_thw: mx.array) -> mx.array:
        """Compute 2D rotary position embeddings from grid dimensions."""
        pos_ids = []
        for t, h, w in grid_thw:
            h, w = int(h), int(w)
            hpos = mx.expand_dims(mx.arange(h), 1)
            hpos = mx.repeat(hpos, w, axis=1)
            hpos = hpos.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            )
            hpos = mx.transpose(hpos, (0, 2, 1, 3)).flatten()

            wpos = mx.expand_dims(mx.arange(w), 0)
            wpos = mx.repeat(wpos, h, axis=0)
            wpos = wpos.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            )
            wpos = mx.transpose(wpos, (0, 2, 1, 3)).flatten()

            stacked = mx.stack([hpos, wpos], axis=-1)
            pos_ids.append(mx.tile(stacked, (int(t), 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(grid_thw[:, 1:])
        rotary_emb = self.rotary_pos_emb(max_grid_size)
        return rotary_emb[pos_ids].reshape(pos_ids.shape[0], -1)

    def _compute_cu_seqlens(self, grid_thw: mx.array) -> mx.array:
        """Compute cumulative sequence lengths for block-diagonal attention."""
        cu_seqlens = []
        for i in range(grid_thw.shape[0]):
            seq_len = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_len, grid_thw[i, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens)
        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        return mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb = self._compute_rotary_pos_emb(grid_thw)
        cu_seqlens = self._compute_cu_seqlens(grid_thw)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize vision weights (Conv3d transpose if needed)."""
        sanitized = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            elif "patch_embed.proj.weight" in k:
                # PyTorch Conv3d: (out, in, T, H, W) -> MLX: (out, T, H, W, in)
                if v.ndim == 5 and v.shape[-1] != self.config.in_channels:
                    v = v.transpose(0, 2, 3, 4, 1)
                sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized
