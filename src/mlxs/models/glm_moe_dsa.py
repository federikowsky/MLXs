"""GLM MoE DSA model: same architecture as DeepSeek V3.2, config via rope_parameters.

Implements ModelProtocol by reusing DeepSeek V3.2 Model with a dedicated ModelArgs
that accepts configs using rope_parameters (e.g. from HuggingFace).

Note: This module imports Model from deepseek_v32 (same architecture, config variant).
If the project later enforces strict no model→model imports, refactor to a
registry alias that uses deepseek_v32.Model with ModelArgs from this module only.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from mlxs.models.base import BaseModelArgs
from mlxs.models.deepseek_v32 import Model as DeepseekV32Model  # same architecture


@dataclass
class ModelArgs(BaseModelArgs):
    """GLM MoE DSA config; compatible with DeepSeek V3.2 layers.

    Supports configs that provide rope_parameters (e.g. rope_theta) instead of
    top-level rope_theta/rope_scaling. Use from_dict to load from such configs.
    """

    model_type: str = "glm_moe_dsa"
    vocab_size: int = 102400
    hidden_size: int = 4096
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: int | None = None
    n_routed_experts: int | None = None
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 1
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        """Build args from config; map rope_parameters to rope_scaling/rope_theta."""
        if "rope_parameters" in params:
            rp = params["rope_parameters"]
            params = {
                **params,
                "rope_scaling": rp,
                "rope_theta": rp.get("rope_theta", 10000.0),
            }
        sig = inspect.signature(cls)
        return cls(**{k: v for k, v in params.items() if k in sig.parameters})


class Model(DeepseekV32Model):
    """GLM MoE DSA: DeepSeek V3.2 backbone with GLM-style config (ModelProtocol)."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
