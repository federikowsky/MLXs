"""MiniMax model architecture — MoE with sigmoid gating and optional QK-norm (§7).

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Uses RoPE, GQA, optional QK-norm, and SwitchGLU MoE with e_score_correction_bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """MiniMax model configuration."""

    model_type: str = "minimax"
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    max_position_embeddings: int = 32768
    num_experts_per_tok: int = 1
    num_local_experts: int = 8
    shared_intermediate_size: int = 0
    num_hidden_layers: int = 24
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rotary_dim: int = 128
    vocab_size: int = 151936
    tie_word_embeddings: bool = False
    scoring_func: str = "sigmoid"
    head_dim: int | None = None
    use_qk_norm: bool = True


class MiniMaxAttention(nn.Module):
    """Multi-head attention with GQA, RoPE, and optional QK-norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        head_dim = args.head_dim or hidden_size // args.num_attention_heads
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.use_qk_norm = getattr(args, "use_qk_norm", True)

        self.q_proj = nn.Linear(
            hidden_size, self.num_attention_heads * head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * head_dim, hidden_size, bias=False
        )
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(
                head_dim * self.num_attention_heads, eps=args.rms_norm_eps
            )
            self.k_norm = nn.RMSNorm(
                head_dim * self.num_key_value_heads, eps=args.rms_norm_eps
            )
        self.rope = nn.RoPE(
            args.rotary_dim, traditional=False, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
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
        return self.o_proj(output)


class MiniMaxSparseMoeBlock(nn.Module):
    """Sparse MoE block with sigmoid gating and e_score_correction_bias."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            bias=False,
        )
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        B, L, H = x.shape
        gates = self.gate(x.astype(mx.float32))
        orig_scores = mx.sigmoid(gates)
        scores = orig_scores + self.e_score_correction_bias
        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        selected = mx.take_along_axis(orig_scores, inds, axis=-1)
        selected = selected / (
            mx.sum(selected, axis=-1, keepdims=True) + 1e-20
        )
        selected = selected.astype(x.dtype)
        x_rep = mx.broadcast_to(
            x[:, :, None, :], (B, L, k, H)
        ).reshape(B * L * k, H)
        inds_flat = inds.reshape(B * L * k)
        y = self.switch_mlp(x_rep, inds_flat)
        n_tok = B * L * k
        y = y.reshape(n_tok, -1)
        if y.shape[-1] != H:
            y = y[:, :H]
        y = y.reshape(B, L, k, H)
        return (y * selected[..., None]).sum(axis=-2)


class MiniMaxDecoderLayer(nn.Module):
    """Pre-norm decoder layer: attention + sparse MoE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = MiniMaxAttention(args)
        self.block_sparse_moe = MiniMaxSparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return r + self.block_sparse_moe(self.post_attention_layernorm(r))


class MiniMaxModel(nn.Module):
    """MiniMax transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MiniMaxDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    """MiniMax LM head wrapper — satisfies ModelProtocol (§7.3, AC17)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache, mask=mask)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Dequantize FP8 weights and restructure MoE expert weights."""

        def dequant(weight: mx.array, scale_inv: mx.array) -> mx.array:
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (
                weight * scale_inv[:, None, :, None]
            ).reshape(m + pad_bottom, n + pad_side)
            return weight[:m, :n].astype(mx.bfloat16)

        new_weights: dict[str, Any] = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                new_weights[wk] = dequant(weight, scale_inv)
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        prefix_check = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        if prefix_check not in weights:
            return weights

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig_name, new_name in mapping.items():
                key = f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight"
                if key not in weights:
                    continue
                to_join = [
                    weights.pop(
                        f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                    )
                    for e in range(self.args.num_local_experts)
                ]
                weights[
                    f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                ] = mx.stack(to_join)
        return weights
