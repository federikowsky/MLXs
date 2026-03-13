"""RWKV7 recurrent model — port from mlx_lm, ModelProtocol-compliant.

Recurrent architecture: per-layer state in ArraysCache(size=3): [0] time-mixing
token shift, [1] WKV7 recurrence state (B, H, D, D), [2] channel-mixing token
shift. Same cache layout as mlx_lm.models.rwkv7 (ArraysCache per layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.layers.activations import relu_squared
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Args for RWKV7; from_dict aligned to config.json (mlx_lm)."""

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    norm_eps: float
    head_dim: int
    num_hidden_layers: int
    a_low_rank_dim: int
    v_low_rank_dim: int
    gate_low_rank_dim: int
    decay_low_rank_dim: int
    tie_word_embeddings: bool = False


def _addcmul(x: mx.array, y: mx.array, z: mx.array) -> mx.array:
    return x + y * z


def _l2_norm(x: mx.array) -> mx.array:
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-7)


def _wkv7_step_ops(
    r: mx.array,
    w: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Single WKV7 step: sab = (state @ a) @ b; state = state*w + v@k + sab; y = state @ r."""
    sab = (state @ a[..., None]) @ b[..., None, :]
    state = state * w[:, :, None, :] + v[..., None] @ k[..., None, :] + sab
    y = state @ r[..., None]
    return y, state


class LayerNormPerHead(nn.Module):
    """Per-head layer norm (RWKV7-specific). Not in layers/norms: used only here."""

    def __init__(self, head_dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.zeros((num_heads, head_dim))
        self.bias = mx.zeros((num_heads, head_dim))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps) + self.bias


class LoRA(nn.Module):
    """Low-rank adapter: linear -> activation -> linear (used for decay, v, a, gate)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool = True,
        activation: str | None = "tanh",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias
        if activation is None:
            self.activation = nn.Identity()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.lora = [
            nn.Linear(self.input_dim, self.low_rank_dim, bias=False),
            self.activation,
            nn.Linear(self.low_rank_dim, self.output_dim, bias=self.bias),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return self.lora[2](self.lora[1](self.lora[0](x)))


class TokenShift(nn.Module):
    """Causal token shift: prev token for recurrence; state is last token from cache."""

    def __call__(
        self,
        x: mx.array,
        state: mx.array | None,
    ) -> mx.array:
        B, L, D = x.shape
        if state is None:
            state = mx.zeros((B, 1, D), x.dtype)
        if L == 1:
            return state
        return mx.concatenate([state, x[:, :-1, :]], axis=1)


class Rwkv7ChannelMixing(nn.Module):
    """Channel mixing: key/value MLP with token shift and time-mix (x_k)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.key = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.value = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.x_k = mx.zeros((args.hidden_size,))
        self.token_shift = TokenShift()

    def __call__(
        self,
        x: mx.array,
        cache: ArraysCache | None,
    ) -> mx.array:
        state = cache[2] if cache is not None else None
        x_prev = self.token_shift(x, state)
        xx = _addcmul(x, x_prev - x, self.x_k)
        if cache is not None:
            cache[2] = x[:, -1:, :]
        return self.value(relu_squared(self.key(xx)))


class Rwkv7TimeMixing(nn.Module):
    """WKV7 time mixing: receptance/decay/key/value with LoRAs and per-head norm.

    Cache layout: cache[0] = token shift (last x), cache[1] = WKV state (B, H, D, D).
    """

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.args = args
        self.hidden_size = args.hidden_size
        self.head_dim = args.head_dim
        self.num_heads = self.hidden_size // self.head_dim
        self.a_low_rank_dim = args.a_low_rank_dim
        self.v_low_rank_dim = args.v_low_rank_dim
        self.gate_low_rank_dim = args.gate_low_rank_dim
        self.decay_low_rank_dim = args.decay_low_rank_dim
        self.token_shift = TokenShift()

        self.x_r = mx.zeros((1, 1, self.hidden_size))
        self.x_w = mx.zeros((1, 1, self.hidden_size))
        self.x_k = mx.zeros((1, 1, self.hidden_size))
        self.x_v = mx.zeros((1, 1, self.hidden_size))
        self.x_a = mx.zeros((1, 1, self.hidden_size))
        self.x_g = mx.zeros((1, 1, self.hidden_size))
        self.k_k = mx.zeros((self.num_heads, self.head_dim))
        self.k_a = mx.zeros((self.num_heads, self.head_dim))
        self.r_k = mx.zeros((self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.g_norm = LayerNormPerHead(self.head_dim, self.num_heads, eps=64e-5)
        self.w_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.decay_low_rank_dim,
            activation="tanh",
        )
        self.v_lora: LoRA | None = None
        if self.layer_idx > 0:
            self.v_lora = LoRA(
                self.hidden_size,
                self.hidden_size,
                low_rank_dim=self.v_low_rank_dim,
                activation=None,
            )
        self.a_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.a_low_rank_dim,
            activation=None,
        )
        self.g_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.gate_low_rank_dim,
            activation="sigmoid",
            bias=False,
        )

    def _wkv7(
        self,
        r: mx.array,
        w: mx.array,
        k: mx.array,
        v: mx.array,
        a: mx.array,
        b: mx.array,
        state: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        B, L, _, _ = r.shape
        if state is None:
            state = mx.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=r.dtype)
        ys: list[mx.array] = []
        for t in range(L):
            y, state = _wkv7_step_ops(r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], state)
            ys.append(y)
        y = mx.stack(ys, axis=1).astype(r.dtype)
        return y, state

    def __call__(
        self,
        x: mx.array,
        v_first: mx.array | None,
        cache: ArraysCache | None,
    ) -> tuple[mx.array, mx.array | None]:
        if cache is None:
            token_shift_cache, state_cache = None, None
        else:
            token_shift_cache, state_cache = cache[0], cache[1]
        B, L, D = x.shape
        x_prev = self.token_shift(x, token_shift_cache)
        xx = x_prev - x

        xr = _addcmul(x, xx, self.x_r)
        xw = _addcmul(x, xx, self.x_w)
        xk = _addcmul(x, xx, self.x_k)
        xv = _addcmul(x, xx, self.x_v)
        xa = _addcmul(x, xx, self.x_a)
        xg = _addcmul(x, xx, self.x_g)

        key = self.k_proj(xk).reshape(B, L, self.num_heads, self.head_dim)
        value = self.v_proj(xv).reshape(B, L, self.num_heads, self.head_dim)
        receptance = self.r_proj(xr).reshape(B, L, self.num_heads, self.head_dim)
        iclr = mx.sigmoid(self.a_lora(xa)).reshape(B, L, self.num_heads, self.head_dim)
        gate = self.g_lora(xg)

        if self.layer_idx == 0:
            v_first = value
        else:
            v_lora = self.v_lora
            assert v_lora is not None
            vv = mx.sigmoid(v_lora(xv)).reshape(B, L, self.num_heads, self.head_dim)
            value = _addcmul(value, v_first - value, vv)

        decay = mx.sigmoid(self.w_lora(xw).reshape(B, L, self.num_heads, self.head_dim)).astype(
            mx.float32
        )
        decay = mx.exp(-0.606531 * decay).astype(receptance.dtype)
        kk = _l2_norm(key * self.k_k)
        key = key * (1 + (iclr - 1) * self.k_a)
        a_arr = -kk
        b_arr = kk * iclr

        out, new_state = self._wkv7(receptance, decay, key, value, a_arr, b_arr, state_cache)
        out = self.g_norm(out.reshape(B, L, self.num_heads, self.head_dim))
        out = (out + (receptance * key * self.r_k).sum(axis=-1, keepdims=True) * value).reshape(
            [B, L, D]
        )
        if cache is not None:
            cache[0] = x[:, -1:, :]
            cache[1] = new_state
        return self.o_proj(out * gate), v_first


class Rwkv7Layer(nn.Module):
    """One RWKV7 block: optional pre_norm (layer 0), time mixing, ffn (channel mixing)."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        if self.layer_idx == 0:
            self.pre_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)
        self.attn = Rwkv7TimeMixing(args, layer_idx=self.layer_idx)
        self.ffn = Rwkv7ChannelMixing(args)
        self.attn_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        v_first: mx.array | None,
        cache: ArraysCache | None,
    ) -> tuple[mx.array, mx.array | None]:
        if self.layer_idx == 0:
            x = self.pre_norm(x)
        h, v_first = self.attn(self.attn_norm(x), v_first, cache)
        h = x + h
        out = h + self.ffn(self.ffn_norm(h), cache)
        return out, v_first


class Rwkv7Model(nn.Module):
    """RWKV7 backbone: embed -> layers (each with ArraysCache size 3) -> norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Rwkv7Layer(args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: list[ArraysCache] | None,
    ) -> mx.array:
        x = self.embeddings(x)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        v_first: mx.array | None = None
        for layer, c in zip(self.layers, cache, strict=True):
            x, v_first = layer(x, v_first, c)
        return self.norm(x)


class Model(nn.Module):
    """RWKV7 LM head wrapper — satisfies ModelProtocol.

    make_cache returns one ArraysCache(size=3) per layer (mlx_lm design):
    [0] time-mixing token shift, [1] WKV state, [2] channel-mixing token shift.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Rwkv7Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[ArraysCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        x = self.model(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            logits = self.model.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)
        return logits

    def make_cache(self) -> list[ArraysCache]:
        """One ArraysCache(size=3) per layer — same as mlx_lm.models.rwkv7.Model."""
        return [ArraysCache(size=3) for _ in range(len(self.model.layers))]

    @property
    def num_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Reshape k_k, k_a, g_norm from (hidden_size,) to (num_heads, head_dim)."""
        for k, v in list(weights.items()):
            if "k_k" in k or "k_a" in k or "g_norm" in k:
                weights[k] = v.reshape(
                    self.args.hidden_size // self.args.head_dim,
                    self.args.head_dim,
                )
        return weights
