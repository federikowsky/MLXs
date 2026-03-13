"""GraniteMoE Hybrid model (Mamba2 + attention, optional MoE) — port from mlx_lm.

Implements ModelProtocol. Alternating Mamba and attention layers via layer_types,
with optional MoE + shared MLP or dense MLP. Cache: KVCache for attention,
ArraysCache(size=2) for Mamba. Compatible with mlx_lm-converted weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.layers.ssm import ssm_update
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """GraniteMoE Hybrid config; from_dict maps config.json (same as mlx_lm/HF)."""

    model_type: str = "granitemoehybrid"
    vocab_size: int = 49152
    hidden_size: int = 2048
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    max_position_embeddings: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    attention_bias: bool = False
    embedding_multiplier: float = 1.0
    attention_multiplier: float = 1.0
    logits_scaling: float = 1.0
    residual_multiplier: float = 1.0
    layer_types: list[str] = field(default_factory=lambda: ["attention", "mamba"])
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    position_embedding_type: str = "rope"
    tie_word_embeddings: bool = True
    time_step_limit: tuple[float, float] = (0.001, 100.0)
    # MoE (optional)
    num_local_experts: int | None = None
    num_experts_per_tok: int | None = None
    shared_intermediate_size: int | None = None
    # Mamba2 (optional for non-hybrid)
    mamba_n_heads: int = 64
    mamba_d_head: int = 64
    mamba_proj_bias: bool = False
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_n_groups: int = 1
    mamba_conv_bias: bool = False
    # Dense MLP (when not MoE)
    mlp_bias: bool = False

    @property
    def use_moe(self) -> bool:
        return bool(self.num_local_experts)


class RMSNormGated(nn.Module):
    """RMSNorm with optional SwiGLU gate (Mamba2-style). Used only in Mamba mixer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        gate: mx.array | None = None,
    ) -> mx.array:
        if gate is not None:
            hidden_states = swiglu(gate, hidden_states)
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


class GraniteMoeHybridMamba2Mixer(nn.Module):
    """Mamba2-style SSM mixer with gated RMSNorm (GraniteMoE Hybrid)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_heads = args.mamba_n_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_n_heads * args.mamba_d_head
        self.n_groups = args.mamba_n_groups
        self.head_dim = args.mamba_d_head
        self.time_step_limit = args.time_step_limit

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.mamba_d_conv,
            padding=0,
            groups=self.conv_dim,
            bias=args.mamba_conv_bias,
        )
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.mamba_proj_bias)
        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)
        self.norm = RMSNormGated(self.intermediate_size, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

    def _conv(
        self,
        conv_input: mx.array,
        cache: ArraysCache | None,
        mask: mx.array | None,
    ) -> mx.array:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)
        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)])
        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: ArraysCache | None,
        mask: mx.array | None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        state = cache[1] if cache else None
        lengths = cache.lengths if cache else None
        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D.astype(hidden_states.dtype),
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask=mask,
            lengths=lengths,
        )
        if cache is not None:
            cache[1] = state
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        projected = self.in_proj(hidden_states)
        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._conv(conv_input, cache, mask)
        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y = self._ssm(hidden_states_ssm, B, C, dt, cache, mask)
        if cache is not None:
            cache.advance(y.shape[1])
        y = self.norm(y, gate)
        return self.out_proj(y)


class GraniteMoeHybridAttention(nn.Module):
    """Multi-head attention with optional RoPE (GraniteMoE Hybrid)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = args.attention_multiplier
        bias = args.attention_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)
        use_rope = args.position_embedding_type != "nope"
        if use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=None,
                max_position_embeddings=args.max_position_embeddings,
            )
        else:
            self.rope = None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.rope is not None:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class GraniteMoeHybridTopKGating(nn.Module):
    """Top-k expert gating for GraniteMoE Hybrid MoE."""

    def __init__(self, input_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.layer(hidden_states)
        top_k_idx = mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        top_k_logits = mx.take_along_axis(logits, top_k_idx, axis=-1)
        top_k_gates = mx.softmax(top_k_logits.astype(mx.float32), axis=-1).astype(logits.dtype)
        return top_k_idx, top_k_gates


class GraniteMoeHybridMoE(nn.Module):
    """MoE block with top-k gating and SwitchGLU experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.input_size = args.hidden_size
        self.hidden_size = args.intermediate_size
        self.switch_mlp = SwitchGLU(self.input_size, self.hidden_size, args.num_local_experts)
        self.router = GraniteMoeHybridTopKGating(
            input_size=self.input_size,
            num_experts=args.num_local_experts,
            top_k=args.num_experts_per_tok,
        )

    def __call__(self, x: mx.array) -> mx.array:
        token_ids, gates = self.router(x)
        y = self.switch_mlp(x, token_ids)
        return (y * gates[..., None]).sum(axis=-2).astype(y.dtype)


class GraniteMoeHybridSharedMLP(nn.Module):
    """Shared SiGLU MLP (MoE mode)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        shared = args.shared_intermediate_size or args.intermediate_size
        self.input_linear = nn.Linear(args.hidden_size, shared * 2, bias=False)
        self.output_linear = nn.Linear(shared, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.input_linear(x), 2, axis=-1)
        return self.output_linear(swiglu(gate, up))


class GraniteMoeHybridMLP(nn.Module):
    """Dense SiGLU MLP (non-MoE mode)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        bias = args.mlp_bias
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class GraniteMoeHybridLayer(nn.Module):
    """Single hybrid layer: Mamba or Attention, then MoE+shared MLP or dense MLP."""

    def __init__(self, args: ModelArgs, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.residual_multiplier = args.residual_multiplier
        self.use_moe = args.use_moe
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if layer_type == "mamba":
            self.mamba = GraniteMoeHybridMamba2Mixer(args)
        elif layer_type == "attention":
            self.self_attn = GraniteMoeHybridAttention(args)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        if self.use_moe:
            self.shared_mlp = GraniteMoeHybridSharedMLP(args)
            self.block_sparse_moe = GraniteMoeHybridMoE(args)
        else:
            self.mlp = GraniteMoeHybridMLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        residual = x
        hidden_states = self.input_layernorm(x)
        if self.layer_type == "mamba":
            hidden_states = self.mamba(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)
        hidden_states = residual + hidden_states * self.residual_multiplier
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        if self.use_moe:
            moe_out = self.block_sparse_moe(normed)
            shared_out = self.shared_mlp(normed)
            mlp_out = moe_out + shared_out
        else:
            mlp_out = self.mlp(normed)
        return residual + mlp_out * self.residual_multiplier


class GraniteMoeHybridModel(nn.Module):
    """GraniteMoE Hybrid backbone: embed + layers + norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [GraniteMoeHybridLayer(args, lt) for lt in args.layer_types]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier
        self._attn_idx = (
            args.layer_types.index("attention") if "attention" in args.layer_types else None
        )
        self._ssm_idx = args.layer_types.index("mamba") if "mamba" in args.layer_types else None

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs) * self.embedding_multiplier
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        attn_mask = (
            create_attention_mask(hidden_states, cache[self._attn_idx])
            if self._attn_idx is not None
            else None
        )
        ssm_mask = (
            create_ssm_mask(hidden_states, cache[self._ssm_idx])
            if self._ssm_idx is not None
            else None
        )
        for layer, c in zip(self.layers, cache, strict=True):
            mask = attn_mask if layer.layer_type == "attention" else ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)
        return self.norm(hidden_states)


class Model(nn.Module):
    """GraniteMoE Hybrid LM head wrapper — satisfies ModelProtocol (AC17)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteMoeHybridModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | ArraysCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out / self.logits_scaling

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        caches: list[KVCache | ArraysCache] = []
        for layer in self.model.layers:
            if layer.layer_type == "mamba":
                caches.append(ArraysCache(size=2))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        if self.args.use_moe and any("block_sparse_moe.input_linear.weight" in k for k in weights):
            num_layers = len(self.args.layer_types)
            for layer_idx in range(num_layers):
                prefix = f"model.layers.{layer_idx}.block_sparse_moe"
                key = f"{prefix}.input_linear.weight"
                if key not in weights:
                    continue
                input_weight = weights.pop(key)
                _, expert_hidden, _ = input_weight.shape
                gate_proj = input_weight[:, : expert_hidden // 2, :]
                up_proj = input_weight[:, expert_hidden // 2 :, :]
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_proj
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_proj
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.output_linear.weight"
                )
        elif not self.args.use_moe and any("shared_mlp.input_linear.weight" in k for k in weights):
            num_layers = len(self.args.layer_types)
            for layer_idx in range(num_layers):
                prefix = f"model.layers.{layer_idx}.shared_mlp"
                key = f"{prefix}.input_linear.weight"
                if key not in weights:
                    continue
                input_weight = weights.pop(key)
                gate_proj, up_proj = mx.split(input_weight, 2, axis=0)
                weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
                weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
                weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.output_linear.weight"
                )
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def quant_predicate(self) -> Any:
        def predicate(path: str, _: Any) -> dict[str, int] | bool:
            if self.args.use_moe and path.endswith("router.layer"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self) -> list[GraniteMoeHybridLayer]:
        return self.model.layers
