"""Reusable model building blocks — activations, RoPE, attention, SSM, MoE, MLA.

Used by model architectures in models/. No dependency on models or cache.
"""

from mlxs.layers.activations import XieLU, swiglu, xielu
from mlxs.layers.attention import (
    quantized_scaled_dot_product_attention,
    scaled_dot_product_attention,
)
from mlxs.layers.mla import MultiLinear, QuantizedMultiLinear
from mlxs.layers.moe import SwitchGLU, SwitchLinear, SwitchMLP
from mlxs.layers.rope import (
    Llama3RoPE,
    SuScaledRoPE,
    YarnRoPE,
    initialize_rope,
)
from mlxs.layers.ssm import (
    compute_dt,
    ssm_attn,
    ssm_update,
    ssm_update_kernel,
)

__all__ = [
    "Llama3RoPE",
    "MultiLinear",
    "QuantizedMultiLinear",
    "SuScaledRoPE",
    "SwitchGLU",
    "SwitchLinear",
    "SwitchMLP",
    "XieLU",
    "YarnRoPE",
    "compute_dt",
    "initialize_rope",
    "quantized_scaled_dot_product_attention",
    "scaled_dot_product_attention",
    "ssm_attn",
    "ssm_update",
    "ssm_update_kernel",
    "swiglu",
    "xielu",
]
