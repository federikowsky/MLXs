"""Nemotron-H hybrid (Mamba2 + attention + MLP + MoE) — port from mlx_lm, ModelProtocol-compliant.

Block types from hybrid_override_pattern: M (Mamba2), * (attention), - (MLP), E (MoE).
RMSNorm, Mamba2 group RMSNorm+gated, ReLU2 MLP, SwitchMLP MoE with ReLU2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchMLP
from mlxs.layers.ssm import ssm_update
from mlxs.models.base import BaseModelArgs

# ----- ModelArgs -----


@dataclass
class ModelArgs(BaseModelArgs):
    """Nemotron-H config (config.json)."""

    model_type: str = "nemotron_h"
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    max_position_embeddings: int = 8192
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    attention_bias: bool = False
    mamba_num_heads: int = 16
    mamba_head_dim: int = 64
    mamba_proj_bias: bool = False
    ssm_state_size: int = 16
    conv_kernel: int = 4
    n_groups: int = 1
    mlp_bias: bool = False
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = False
    use_conv_bias: bool = False
    hybrid_override_pattern: list[str] = ()
    head_dim: int | None = None
    moe_intermediate_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    n_group: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    topk_group: int | None = None
    num_experts_per_tok: int | None = None
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    time_step_limit: tuple[float, float] | None = None
    time_step_min: float | None = None
    time_step_max: float | None = None

    def __post_init__(self) -> None:
        if self.hybrid_override_pattern is None:
            self.hybrid_override_pattern = []
        if (
            self.time_step_limit is None
            and self.time_step_min is not None
            and self.time_step_max is not None
        ):
            self.time_step_limit = (self.time_step_min, self.time_step_max)
        if self.time_step_limit is None:
            self.time_step_limit = (0.001, 100.0)


# ----- Mamba RMSNorm (gated, group) — model-specific -----


class MambaRMSNormGated(nn.Module):
    """Group RMSNorm with optional SwiGLU gate (Nemotron-H Mamba2)."""

    def __init__(self, hidden_size: int, eps: float, group_size: int) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,))
        self.group_size = group_size

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        if gate is not None:
            x = swiglu(gate, x)
        x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * x.flatten(-2)


# ----- Mamba2 mixer -----


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_heads = args.mamba_num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.mamba_num_heads * args.mamba_head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.mamba_head_dim
        self.time_step_limit = args.time_step_limit or (0.001, 100.0)
        self.heads_per_group = self.num_heads // self.n_groups

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.mamba_proj_bias)

        self.dt_bias = mx.ones((self.num_heads,))
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones((self.num_heads,))

        group_size = self.intermediate_size // self.n_groups
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=args.layer_norm_epsilon,
            group_size=group_size,
        )
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
            conv_input = mx.where(mask[..., None], conv_input, 0.0)

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
        state = cache[1] if cache is not None else None
        lengths = cache.lengths if cache is not None else None

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


# ----- Attention -----


class NemotronHAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = (
            args.head_dim
            if args.head_dim is not None
            else (args.hidden_size // args.num_attention_heads)
        )
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# ----- MLP (ReLU2) -----


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int | None = None) -> None:
        super().__init__()
        inter = intermediate_size or args.intermediate_size
        self.up_proj = nn.Linear(args.hidden_size, inter, bias=args.mlp_bias)
        self.down_proj = nn.Linear(inter, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.relu2(self.up_proj(x)))


# ----- MoE gate (group expert select) -----


def _group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    orig_scores = scores = mx.sigmoid(gates.astype(mx.float32))
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok or 1
        self.norm_topk_prob = config.norm_topk_prob or False
        self.n_routed_experts = config.n_routed_experts or 0
        self.routed_scaling_factor = config.routed_scaling_factor or 1.0
        self.n_group = config.n_group or 1
        self.topk_group = config.topk_group or 0
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return _group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class NemotronHMoE(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok or 1
        self.switch_mlp = SwitchMLP(
            config.hidden_size,
            config.moe_intermediate_size or config.intermediate_size,
            config.n_routed_experts or 1,
            activation=nn.ReLU2(),
        )

        self.gate = MoEGate(config)
        self.shared_experts: NemotronHMLP | None = None
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            inter = config.moe_shared_expert_intermediate_size or config.intermediate_size
            self.shared_experts = NemotronHMLP(config, intermediate_size=inter)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


# ----- Block -----


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelArgs, block_type: str) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.block_type = block_type

        if block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args)
        elif block_type == "*":
            self.mixer = NemotronHAttention(args)
        elif block_type == "-":
            self.mixer = NemotronHMLP(args)
        elif block_type == "E":
            self.mixer = NemotronHMoE(args)
        else:
            raise ValueError(f"Unknown block_type: {block_type!r}")

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: ArraysCache | KVCache | None = None,
    ) -> mx.array:
        hidden_states = self.norm(x)
        if self.block_type in ("M", "*"):
            hidden_states = self.mixer(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.mixer(hidden_states)
        return x + hidden_states


# ----- Backbone -----


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        pattern = args.hybrid_override_pattern or []
        self.layers = [NemotronHBlock(args, bt) for bt in pattern]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self._fa_idx = 0
        self._ssm_idx = 0
        for b in pattern:
            if b == "*":
                break
            if b == "M":
                self._fa_idx += 1
        for b in pattern:
            if b == "*":
                self._ssm_idx += 1
            elif b == "M":
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: list[ArraysCache | KVCache] | None = None,
    ) -> mx.array:
        hidden_states = self.embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        attn_mask = create_attention_mask(hidden_states, cache[self._fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self._ssm_idx])

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type in ("M", "*"):
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            mask = attn_mask if layer.block_type == "*" else ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm_f(hidden_states)


# ----- Model (ModelProtocol) -----


class Model(nn.Module):
    """Nemotron-H LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[ArraysCache | KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.backbone(input_ids, cache=cache)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.backbone.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[ArraysCache | KVCache]:
        caches: list[ArraysCache | KVCache] = []
        for layer in self.backbone.layers:
            if layer.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif layer.block_type == "*":
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        num_hidden_layers = len(self.args.hybrid_override_pattern or [])
        for layer_idx in range(num_hidden_layers):
            prefix = f"backbone.layers.{layer_idx}.mixer"
            for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                key = f"{prefix}.experts.0.{m}.weight"
                if key in weights:
                    n_experts = self.args.n_routed_experts or 1
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{m}.weight") for e in range(n_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        return weights
