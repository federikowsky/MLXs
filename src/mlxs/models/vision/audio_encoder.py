"""Gemma 3N audio encoder — conformer-based mel-spectrogram encoder.

Ported from mlx_vlm/models/gemma3n/audio.py + gemma3n.py (MultimodalEmbedder).
Self-contained: no dependencies on mlx_vlm. Uses only mlx.core, mlx.nn,
and stdlib.

Key classes:
- AudioConfig — dataclass holding all conformer / sub-sample-conv parameters.
- AudioModel — the main encoder (sub-sample conv projection + conformer stack).
- Gemma3nMultimodalEmbedder — projects audio (or hard-token) features into
  the language-model embedding space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlxs.models.base import BaseModelArgs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig(BaseModelArgs):
    """Configuration for the Gemma 3N audio encoder.

    Fields mirror the upstream ``gemma3n`` config.json ``audio_config`` block.
    """

    input_feat_size: int = 80
    hidden_size: int = 1536
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_invalid_logits_value: float = -1e9
    conf_attention_logit_cap: float = 50.0
    conf_num_attention_heads: int = 8
    conf_num_hidden_layers: int = 12
    conf_conv_kernel_size: int = 5
    conf_positional_bias_size: int = 256
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    sscp_conv_channel_size: tuple[int, int] = (128, 32)
    sscp_conv_group_norm_eps: float = 1e-3
    sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = (
        (3, 3),
        (3, 3),
    )
    sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = (
        (2, 2),
        (2, 2),
    )
    vocab_size: int = 128
    sscp_conv_eps: float = 1e-3
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 10_000_000_000.0
    vocab_offset: int = 262_144 + 128  # text_vocab_size + vision_vocab_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_array_shape(arr: mx.array) -> bool:
    """Return True when conv weight is already in MLX layout (no transpose needed)."""
    shape = arr.shape
    if len(shape) == 4:
        out_channels, kH, kW, _ = shape
        return out_channels >= kH and out_channels >= kW and kH == kW
    if len(shape) == 3:
        _, kW, out_channels = shape
        return kW >= out_channels
    return False


def _torch_to_mlx_pad_width(
    padding: tuple[int, ...], ndim: int
) -> list[tuple[int, int]]:
    """Convert PyTorch-style padding to ``mx.pad`` *pad_width*.

    PyTorch ``F.pad`` format: ``(left, right, top, bottom, front, back, …)``
    applied from the *last* dimension backwards.  MLX ``mx.pad`` expects a
    list of ``(before, after)`` per dimension starting from dim-0.
    """
    pad_width: list[tuple[int, int]] = [(0, 0)] * ndim
    if ndim >= 1 and len(padding) >= 2:
        pad_width[-1] = (padding[0], padding[1])
    if ndim >= 2 and len(padding) >= 4:
        pad_width[-2] = (padding[2], padding[3])
    if ndim >= 3 and len(padding) >= 6:
        pad_width[-3] = (padding[4], padding[5])
    if ndim >= 4 and len(padding) >= 8:
        pad_width[-4] = (padding[6], padding[7])
    return pad_width


# ---------------------------------------------------------------------------
# RMSNorm (self-contained copy — no dependency on language model)
# ---------------------------------------------------------------------------


class Gemma3nRMSNorm(nn.Module):
    """RMSNorm with optional learnable scale and constant shift."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        scale_shift: float = 0.0,
        with_scale: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.scale_shift = scale_shift
        self.with_scale = with_scale
        self.weight: mx.array | None = mx.ones(dim) if with_scale else None

    def __call__(self, x: mx.array) -> mx.array:
        out = x.astype(mx.float32)
        out = out * mx.rsqrt(out.square().mean(axis=-1, keepdims=True) + self.eps)
        if self.with_scale and self.weight is not None:
            out = out * (self.weight + self.scale_shift)
        return out.astype(x.dtype)


# ---------------------------------------------------------------------------
# Relative position embedding
# ---------------------------------------------------------------------------


class Gemma3nAudioRelativePositionEmbedding(nn.Module):
    """Sinusoidal relative position bias for chunked attention."""

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config

        self.num_heads = config.conf_num_attention_heads
        self.channels = config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = (
            config.conf_attention_context_left - 1
            if config.conf_attention_context_left > 0
            else 0
        )
        self.max_forward = config.conf_attention_context_right

        self.pos_proj = nn.Linear(
            self.channels, self.num_heads * self.head_dim, bias=False
        )

        num_timescales = self.channels // 2
        log_inc = math.log(1.0e4) / max(num_timescales - 1, 1)
        inv_ts = mx.exp(mx.arange(num_timescales) * -log_inc)
        self._inv_timescales = inv_ts[None, None, :]

    # -- internal helpers ---------------------------------------------------

    def _timing_signal(self, position: mx.array, dtype: mx.Dtype) -> mx.array:
        pos = mx.expand_dims(position.astype(mx.float32), axis=-1)
        scaled = pos * self._inv_timescales
        return mx.concatenate([mx.sin(scaled), mx.cos(scaled)], axis=-1).astype(dtype)

    def _relative_shift(
        self,
        term_bd: mx.array,
        batch_size: int,
        num_heads: int,
        num_query_blocks: int,
        query_block_size: int,
        key_context_size: int,
        max_span_plus_1: int,
    ) -> mx.array:
        pad_amount = (key_context_size + 1) - max_span_plus_1
        padded = mx.pad(
            term_bd,
            _torch_to_mlx_pad_width((0, pad_amount), term_bd.ndim),
        )
        reshaped = padded.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size * (key_context_size + 1),
        )
        sliced = reshaped[:, :, :, : query_block_size * key_context_size]
        return sliced.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        )

    # -- forward ------------------------------------------------------------

    def __call__(self, queries: mx.array, keys: mx.array) -> mx.array:
        """Compute content + position attention logits.

        Args:
            queries: ``[B, U, W, N, H]``
            keys:    ``[B, U, C, N, H]``

        Returns:
            Logits ``[B, N, U, W, C]``.
        """
        B, U, W, N, H = queries.shape
        C = keys.shape[2]

        pos_indices = mx.expand_dims(
            mx.arange(self.max_backward, -self.max_forward - 1, -1), axis=0
        )
        F_span = pos_indices.shape[1]

        sin_emb = self._timing_signal(pos_indices, queries.dtype)
        sin_emb = self.pos_proj(sin_emb).reshape(1, F_span, N, H).squeeze(0)

        # Content interaction  [B,N,U,W,C]
        q_p = queries.transpose(0, 3, 1, 2, 4)
        k_p = keys.transpose(0, 3, 1, 4, 2)
        term_ac = q_p @ k_p

        # Position interaction  [B,N,U,W,F]
        q_flat = q_p.reshape(B, N, U * W, H)
        s_t = sin_emb.transpose(1, 2, 0)  # [N,H,F]
        term_bd = (q_flat @ s_t).reshape(B, N, U, W, F_span)

        term_bd = self._relative_shift(term_bd, B, N, U, W, C, F_span)
        return term_ac + term_bd


# ---------------------------------------------------------------------------
# Chunked attention
# ---------------------------------------------------------------------------


class Gemma3nAudioAttention(nn.Module):
    """Chunked local attention with relative-position bias and logit soft-cap."""

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config

        self.num_heads = config.conf_num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = config.conf_attention_chunk_size
        self.max_future_horizon = config.conf_attention_context_right
        self.max_past_horizon = max(0, config.conf_attention_context_left - 1)
        self.attention_invalid_logits_value = config.conf_attention_invalid_logits_value
        self.attention_logits_soft_cap = config.conf_attention_logit_cap
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )

        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(config)
        self.per_dim_scale = mx.zeros((self.head_dim,))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        q_scale = self.head_dim ** -0.5
        r_softplus_0 = 1.0 / mx.log(2.0)
        self._q_scale = q_scale * r_softplus_0

        # Pre-compute local causal mask  [W, C]
        lower = mx.tril(mx.ones((self.context_size, self.chunk_size), dtype=mx.bool_), k=0).T
        upper = mx.tril(
            mx.ones((self.chunk_size, self.context_size), dtype=mx.bool_),
            k=self.max_past_horizon + self.max_future_horizon,
        )
        self._local_causal_valid_mask = mx.ones(
            (self.chunk_size, self.context_size), dtype=mx.bool_
        ) * lower * upper

        self._softcap = mx.array(self.attention_logits_soft_cap, dtype=mx.float32)

    # -- helpers ------------------------------------------------------------

    def _pad_dim1(self, x: mx.array, before: int, after: int) -> mx.array:
        pad_spec = [0] * x.ndim * 2
        idx = 2 * (x.ndim - 2)
        pad_spec[idx] = before
        pad_spec[idx + 1] = after
        return mx.pad(x, _torch_to_mlx_pad_width(tuple(pad_spec), x.ndim))

    def _convert_to_block(
        self, x: mx.array, padding_val: Union[bool, float] = 0.0
    ) -> mx.array:
        b, t, *rest = x.shape
        n_blocks = (t + self.chunk_size - 1) // self.chunk_size
        pad_len = n_blocks * self.chunk_size - t
        if pad_len > 0:
            x = self._pad_dim1(x, 0, pad_len)
        return x.reshape(b, n_blocks, self.chunk_size, *rest)

    @staticmethod
    def _unfold(x: mx.array, dimension: int, size: int, step: int) -> mx.array:
        dim_size = x.shape[dimension]
        n_windows = (dim_size - size) // step + 1
        slices_base = [slice(None)] * len(x.shape)
        windows = []
        for i in range(n_windows):
            s = list(slices_base)
            s[dimension] = slice(i * step, i * step + size)
            windows.append(x[tuple(s)])
        return mx.stack(windows, axis=dimension + 1)

    def _extract_block_context(self, x: mx.array) -> mx.array:
        x = self._pad_dim1(x, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1)
        x_unfolded = self._unfold(x, 1, self.context_size, self.chunk_size)
        if x.ndim > 2 and x_unfolded.ndim > 3:
            x_unfolded = x_unfolded.transpose(0, 2, 1, 3, 4)
        return x_unfolded

    # -- forward ------------------------------------------------------------

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)

        # Per-dim scaling  softplus(per_dim_scale) * base_scale
        per_dim = mx.logaddexp(self.per_dim_scale, 0.0).reshape(1, 1, 1, self.head_dim)
        q = q * self._q_scale * per_dim

        q_blocks = self._convert_to_block(q)         # [B, U, W, N, H]
        k_blocks = self._extract_block_context(k)     # [B, U, C, N, H]
        v_blocks = self._extract_block_context(v)     # [B, U, C, N, H]
        U = q_blocks.shape[1]

        # Build validity mask from input padding mask  --------------------------
        valid = ~mask  # True = valid
        ev = self._extract_block_context(valid).transpose(0, 2, 1)  # [B, U, ...]
        if ev.ndim == 4 and ev.shape[:2] == (B, U) and ev.shape[2] * ev.shape[3] == self.context_size:
            ev = ev.reshape(B, U, self.context_size)
        if ev.shape != (B, U, self.context_size):
            raise ValueError(
                f"Extracted validity mask shape {ev.shape} != "
                f"({B}, {U}, {self.context_size})"
            )
        cond_valid = mx.expand_dims(mx.expand_dims(ev, axis=1), axis=-2)  # [B,1,U,1,C]
        cond_causal = self._local_causal_valid_mask[None, None, None, ...]  # [1,1,1,W,C]
        cond = mx.logical_and(cond_valid, cond_causal)

        # Logits  ---------------------------------------------------------------
        logits = self.relative_position_embedding(q_blocks, k_blocks)
        logits = nn.tanh(logits / self._softcap) * self._softcap
        logits = mx.where(cond, logits, self.attention_invalid_logits_value)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1).astype(v_blocks.dtype)

        # Context vectors  [B, U, W, N, H]  ------------------------------------
        _, N, _, W, C = probs.shape
        H = v_blocks.shape[-1]
        prob_flat = probs.transpose(0, 2, 1, 3, 4).reshape(-1, W, C)
        v_flat = v_blocks.transpose(0, 1, 3, 2, 4).reshape(-1, C, H)
        ctx = (prob_flat @ v_flat).reshape(B, U, N, W, H).transpose(0, 1, 3, 2, 4)
        ctx = ctx.reshape(B, U * self.chunk_size, self.num_heads, self.head_dim)[:, :T]
        return ctx


# ---------------------------------------------------------------------------
# Cumulative group norm
# ---------------------------------------------------------------------------


class Gemma3nCumulativeGroupNorm(nn.Module):
    """Group-norm (num_groups=1) applied cumulatively over time.

    Computes running mean/variance across the time axis so that each step
    is normalised using statistics from itself and all prior valid steps.
    """

    def __init__(
        self,
        num_channels: int,
        feature_dims: tuple[int, ...],
        eps: float = 1e-3,
        use_scale: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps
        self.use_scale = use_scale
        self.use_bias = use_bias

        self.weight: mx.array | None = mx.ones(num_channels) if use_scale else None
        self.bias: mx.array | None = mx.zeros(num_channels) if use_bias else None

        # Reduction over all dims except batch (0) and time (1).
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        expected_suffix = self.feature_dims + (self.num_channels,)
        if x.shape[2:] != expected_suffix:
            raise ValueError(
                f"Input suffix {x.shape[2:]} != expected {expected_suffix}"
            )

        calc = mx.float32
        xc = x.astype(calc)

        if mask is not None:
            suffix_ones = (1,) * len(expected_suffix)
            mc = mask.reshape(mask.shape + suffix_ones).astype(calc)
        else:
            mc = mx.ones_like(xc, dtype=calc)

        xm = xc * mc

        sum_at_t = mx.sum(xm, axis=self.reduction_axes, keepdims=True)
        cum_sum = mx.cumsum(sum_at_t, axis=1)

        cnt_at_t = mx.sum(mc, axis=self.reduction_axes, keepdims=True)
        cum_cnt = mx.clip(mx.cumsum(cnt_at_t, axis=1), 1, None)

        cum_mean = cum_sum / cum_cnt

        sq_diff = (xc - cum_mean) ** 2
        sq_at_t = mx.sum(sq_diff * mc, axis=self.reduction_axes, keepdims=True)
        cum_var = mx.cumsum(sq_at_t, axis=1) / cum_cnt

        out = (xc - cum_mean) * mx.rsqrt(cum_var + self.eps)

        if self.use_scale and self.weight is not None:
            s = self.weight.astype(calc)
            out = out * s.reshape([1] * (x.ndim - 1) + [self.num_channels])

        if self.use_bias and self.bias is not None:
            b = self.bias.astype(calc)
            out = out + b.reshape([1] * (x.ndim - 1) + [self.num_channels])

        return (out * mc).astype(x.dtype)


# ---------------------------------------------------------------------------
# Sub-sample convolution projection
# ---------------------------------------------------------------------------


class Gemma3nAudioSSCPConvBlock(nn.Module):
    """Single Conv2d + cumulative-group-norm + ReLU block."""

    def __init__(
        self,
        idx: int,
        input_freq_dim: int,
        config: AudioConfig,
        manual_padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        super().__init__()
        self.config = config
        self.manual_padding = manual_padding

        in_ch = 1 if idx == 0 else config.sscp_conv_channel_size[idx - 1]
        out_ch = config.sscp_conv_channel_size[idx]
        kh, kw = config.sscp_conv_kernel_size[idx]
        sh, sw = config.sscp_conv_stride_size[idx]

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(0, 0),
            bias=False,
        )

        f_in_padded = input_freq_dim + manual_padding[0] + manual_padding[1]
        f_out = (f_in_padded - kw) // sw + 1

        self.norm = Gemma3nCumulativeGroupNorm(
            num_channels=out_ch,
            feature_dims=(f_out,),
            eps=config.sscp_conv_eps,
            use_scale=True,
            use_bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C_in, T, F]
        xp = mx.pad(x, _torch_to_mlx_pad_width(self.manual_padding, x.ndim))
        # Conv2d expects [B, H, W, C] in MLX
        out = self.conv(xp.transpose(0, 2, 3, 1))
        # norm expects [B, T, F, C]
        out = self.norm(out)
        # back to [B, C, T, F]
        return nn.relu(out.transpose(0, 3, 1, 2))


class Gemma3nAudioSubSampleConvProjection(nn.Module):
    """Two Conv2d blocks + linear projection to hidden_size."""

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config

        current_f = config.input_feat_size
        block_paddings: list[tuple[int, int, int, int]] = []
        f_outs: list[int] = []

        for i in range(2):
            kh, kw = config.sscp_conv_kernel_size[i]
            _, sw = config.sscp_conv_stride_size[i]
            pad_t_top, pad_t_bottom = 0, kh - 1
            pad_f_left, pad_f_right = 1, 1
            block_paddings.append((pad_f_left, pad_f_right, pad_t_top, pad_t_bottom))
            f_padded = current_f + pad_f_left + pad_f_right
            current_f = (f_padded - kw) // sw + 1
            f_outs.append(current_f)

        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            idx=0,
            input_freq_dim=config.input_feat_size,
            config=config,
            manual_padding=block_paddings[0],
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            idx=1,
            input_freq_dim=f_outs[0],
            config=config,
            manual_padding=block_paddings[1],
        )
        final_c = config.sscp_conv_channel_size[-1]
        final_f = f_outs[-1]
        self.input_proj_in_features = final_c * final_f
        self.input_proj_linear = nn.Linear(
            self.input_proj_in_features, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, F_in]
        x = mx.expand_dims(x, 1)  # [B, 1, T, F_in]
        x = self.conv_0(x)
        x = self.conv_1(x)
        b, c, t, f = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, t, f * c)
        return self.input_proj_linear(x)


# ---------------------------------------------------------------------------
# Conformer building blocks
# ---------------------------------------------------------------------------


class Gemma3nAudioConformerAttention(nn.Module):
    """Pre-norm attention + post-norm residual with gradient clipping."""

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config

        head_dim = config.hidden_size // config.conf_num_attention_heads
        self.post_in_features = config.hidden_size

        self._gradient_clipping = mx.array(config.gradient_clipping)

        self.pre_attn_norm = Gemma3nRMSNorm(config.hidden_size)
        self.attn = Gemma3nAudioAttention(config)
        self.post = nn.Linear(self.post_in_features, config.hidden_size, bias=False)
        self.post_norm = Gemma3nRMSNorm(config.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        residual = x
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        x = self.pre_attn_norm(x)
        x = self.attn(x, mask)
        b, t, n, h = x.shape
        x = self.post(x.reshape(b, t, n * h))
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        return residual + self.post_norm(x)


class Gemma3nAudioConformerFeedForward(nn.Module):
    """Pre-norm FFN with SiLU, residual scaled by ``conf_residual_weight``."""

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config

        self._gradient_clipping = mx.array(config.gradient_clipping)

        self.pre_layer_norm = Gemma3nRMSNorm(config.hidden_size)
        self.ffw_layer_1 = nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.ffw_layer_2 = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False)
        self.post_layer_norm = Gemma3nRMSNorm(config.hidden_size)
        self._post_layer_scale = mx.array(config.conf_residual_weight)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        x = self.pre_layer_norm(x)
        x = nn.silu(self.ffw_layer_1(x))
        x = self.ffw_layer_2(x)
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        x = self.post_layer_norm(x)
        return residual + x * self._post_layer_scale


class Gemma3nAudioConformerLightConv1d(nn.Module):
    """Depthwise light-conv with GLU gate and causal padding."""

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config

        self.pre_layer_norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_start = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conf_conv_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        self._gradient_clipping = mx.array(config.gradient_clipping)
        self.conv_norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_end = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.causal_padding = config.conf_conv_kernel_size - 1

    def __call__(self, audio_encodings: mx.array) -> mx.array:
        residual = audio_encodings
        x = self.pre_layer_norm(audio_encodings)
        x = nn.glu(self.linear_start(x), axis=-1)
        # [B, T, D] -> [B, D, T] for causal pad, then back for Conv1d
        xt = x.transpose(0, 2, 1)
        xt = mx.pad(xt, _torch_to_mlx_pad_width((self.causal_padding, 0), xt.ndim))
        x = self.depthwise_conv1d(xt.transpose(0, 2, 1))
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        x = nn.silu(self.conv_norm(x))
        return self.linear_end(x) + residual


class Gemma3nAudioConformerBlock(nn.Module):
    """Single conformer block: FFN -> attention -> light-conv -> FFN -> norm."""

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(config)
        self.attention = Gemma3nAudioConformerAttention(config)
        self.lconv1d = Gemma3nAudioConformerLightConv1d(config)
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(config)
        self._gradient_clipping = mx.array(config.gradient_clipping)
        self.norm = Gemma3nRMSNorm(config.hidden_size)

    def __call__(self, audio_encodings: mx.array, audio_mel_mask: mx.array) -> mx.array:
        x = self.ffw_layer_start(audio_encodings)
        x = self.attention(x, audio_mel_mask)
        valid = (~audio_mel_mask).astype(x.dtype)
        x = self.lconv1d(x * mx.expand_dims(valid, -1))
        x = self.ffw_layer_end(x)
        x = mx.clip(x, -self._gradient_clipping, self._gradient_clipping)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Main audio encoder
# ---------------------------------------------------------------------------


class AudioModel(nn.Module):
    """Gemma 3N audio encoder: sub-sample conv projection + conformer stack.

    Accepts mel-spectrogram features and produces a sequence of hidden vectors
    suitable for projection into the language-model embedding space via
    ``Gemma3nMultimodalEmbedder``.
    """

    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config

        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(config)
        self.conformer = [
            Gemma3nAudioConformerBlock(config)
            for _ in range(config.conf_num_hidden_layers)
        ]

    def __call__(
        self, audio_mel: mx.array, audio_mel_mask: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Encode mel features.

        Args:
            audio_mel: ``[B, T, F]`` mel-spectrogram.
            audio_mel_mask: ``[B, T]`` boolean mask (True = padded / invalid).

        Returns:
            Tuple of ``(encodings, mask)`` after conformer + optional reduction.
        """
        encodings = self.subsample_conv_projection(audio_mel)
        t_sub = encodings.shape[1]

        # Sub-sample the mask to match the downsampled time dimension.
        time_stride = 1
        for sp in self.config.sscp_conv_stride_size:
            time_stride *= sp[0]

        indices = mx.clip(mx.arange(t_sub) * time_stride, a_min=None, a_max=audio_mel_mask.shape[1] - 1)
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:
            indices = mx.broadcast_to(indices[None, :], (audio_mel_mask.shape[0], t_sub))

        current_mask = mx.take_along_axis(audio_mel_mask, indices, axis=1)
        if current_mask.shape[1] != t_sub:
            if current_mask.shape[1] > t_sub:
                current_mask = current_mask[:, :t_sub]
            else:
                pad_n = t_sub - current_mask.shape[1]
                current_mask = mx.pad(
                    current_mask,
                    _torch_to_mlx_pad_width((0, pad_n), current_mask.ndim),
                )

        for block in self.conformer:
            encodings = block(encodings, current_mask)

        if self.config.conf_reduction_factor > 1:
            encodings = encodings[:, :: self.config.conf_reduction_factor]
            current_mask = current_mask[:, :: self.config.conf_reduction_factor]

        # Ensure mask length matches after reduction.
        if current_mask.shape[1] != encodings.shape[1]:
            target = encodings.shape[1]
            if current_mask.shape[1] > target:
                current_mask = current_mask[:, :target]
            else:
                pad_n = target - current_mask.shape[1]
                current_mask = mx.pad(
                    current_mask,
                    _torch_to_mlx_pad_width((0, pad_n), current_mask.ndim),
                )

        encodings = mx.where(current_mask[..., None], 0.0, encodings)
        return encodings, current_mask

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Transpose Conv2d / Conv1d weights from PyTorch to MLX layout."""
        sanitized: dict[str, mx.array] = {}
        for k, v in weights.items():
            if "conv.weight" in k:
                sanitized[k] = v if _check_array_shape(v) else v.transpose(0, 2, 3, 1)
            elif "conv1d.weight" in k:
                sanitized[k] = v if _check_array_shape(v) else v.transpose(0, 2, 1)
            else:
                sanitized[k] = v
        return sanitized


# ---------------------------------------------------------------------------
# Multimodal embedder (projects audio features into LM space)
# ---------------------------------------------------------------------------


class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds hard token ids *or* soft (continuous) features into LM space.

    Used for both vision and audio modalities in Gemma 3N.  For audio, the
    ``AudioModel`` output is passed as ``inputs_embeds`` (soft tokens); for
    hard audio tokens the ``input_ids`` path is used with the vocab offset.
    """

    def __init__(
        self,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        vocab_size: int,
        vocab_offset: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.multimodal_hidden_size = multimodal_hidden_size
        self.text_hidden_size = text_hidden_size
        self.vocab_size = vocab_size
        self.vocab_offset = vocab_offset
        self.eps = eps

        self.embedding = nn.Embedding(vocab_size, multimodal_hidden_size)
        self.hard_embedding_norm = Gemma3nRMSNorm(multimodal_hidden_size, eps=eps)
        self.soft_embedding_norm = Gemma3nRMSNorm(multimodal_hidden_size, eps=eps)
        self.embedding_projection = nn.Linear(
            multimodal_hidden_size, text_hidden_size, bias=False
        )
        self.embedding_post_projection_norm = Gemma3nRMSNorm(
            text_hidden_size, eps=eps, with_scale=False
        )

    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
    ) -> mx.array:
        """Project either hard token ids or soft embeddings into LM space.

        Exactly one of ``input_ids`` or ``inputs_embeds`` must be provided.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is not None:
            normed = self.soft_embedding_norm(inputs_embeds)
        else:
            normed = self.hard_embedding_norm(self.embedding(input_ids - self.vocab_offset))

        return self.embedding_post_projection_norm(self.embedding_projection(normed))
