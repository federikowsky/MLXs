"""Sparse expert mixture layers — SwitchLinear, SwitchGLU, SwitchMLP.

Used by MoE models (Mixtral, DeepSeek, Qwen MoE, etc.).
Compatible with mlx_lm switch_layers.py API.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlxs.layers.activations import swiglu


def _gather_sort(x: mx.array, indices: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x: mx.array, inv_order: mx.array, shape: tuple | None = None) -> mx.array:
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ) -> None:
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale, high=scale, shape=(num_experts, output_dims, input_dims)
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.freeze()

    @property
    def input_dims(self) -> int:
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(num_experts, output_dims, input_dims)
        )
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self) -> int:
        return self.weight.shape[2]

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(
        self, group_size: int = 64, bits: int = 4, mode: str = "affine"
    ) -> QuantizedSwitchLinear:
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims, output_dims, num_experts, False, group_size, bits, mode=mode
        )
        ql.weight, ql.scales, *biases = mx.quantize(self.weight, group_size, bits, mode=mode)
        ql.biases = biases[0] if biases else None
        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: nn.Module | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation or SwiGLU()

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: nn.Module | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation or nn.GELU(approx="precise")

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
