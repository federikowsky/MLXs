from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from mlxs._types import ModelMode
from mlxs.config.schema import ModelConfig
from mlxs.convert.api import convert_source, verify_output
from mlxs.convert.errors import ConversionVerificationError, UnsupportedRuntimeTargetError
from mlxs.convert.planning import _flatten_parameter_tree
from mlxs.convert.types import ConversionOptions, VerificationMode
from mlxs.load import load_model, load_model_and_tokenizer, load_tokenizer
from mlxs.load.registry import get_model_classes


@dataclass(frozen=True, slots=True)
class RuntimeCase:
    name: str
    runtime_target: str
    macro_template: str
    config: dict[str, Any]
    source_builder: Callable[[dict[str, np.ndarray], dict[str, Any]], dict[str, np.ndarray]]
    multimodal: bool = False
    tensor_compare: str = "exact"
    tensor_atol: float = 0.0


def _make_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "hello": 3,
                "world": 4,
            },
            unk_token="<pad>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    hf.save_pretrained(path)


def _deterministic_array(name: str, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape)) if shape else 1
    base = (sum(ord(c) for c in name) % 37) / 250.0
    value = np.arange(size, dtype=np.float32).reshape(shape if shape else (1,))
    value = value / max(size, 1) / 10.0 + base
    return value.reshape(shape) if shape else np.array(base, dtype=np.float32)


def _build_reference_tensors(
    model_type: str,
    config: dict[str, Any],
    *,
    multimodal: bool,
) -> dict[str, np.ndarray]:
    ModelClass, ModelArgsClass = get_model_classes(model_type)
    args = ModelArgsClass.from_dict(config)
    signature = inspect.signature(ModelClass.__init__)
    if "model_mode" in signature.parameters:
        model = ModelClass(
            args,
            model_mode=ModelMode.MULTIMODAL if multimodal else ModelMode.TEXT,
        )
    else:
        model = ModelClass(args)
    flat = _flatten_parameter_tree(model.parameters())
    return {
        name: _deterministic_array(name, tuple(int(dim) for dim in value.shape))
        for name, value in flat.items()
    }


def _save_fixture(
    directory: Path,
    config: dict[str, Any],
    tensors: dict[str, np.ndarray],
    *,
    tokenizer: bool = True,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    save_file(tensors, str(directory / "model.safetensors"))
    if tokenizer:
        _make_tokenizer(directory)


def _head_dim(config: dict[str, Any]) -> int:
    if "head_dim" in config:
        return int(config["head_dim"])
    hidden_size = int(config.get("hidden_size") or config["text_config"]["hidden_size"])
    num_attention_heads = int(
        config.get("num_attention_heads") or config["text_config"]["num_attention_heads"]
    )
    return hidden_size // num_attention_heads


def _dense_source_with_extras(
    reference: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source = {name: value.copy() for name, value in reference.items()}
    head_dim = _head_dim(config)
    source["model.layers.0.self_attn.rotary_emb.inv_freq"] = _deterministic_array(
        "rotary.inv_freq",
        (max(1, head_dim // 2),),
    )
    if "lm_head.weight" not in source:
        embed_name = next(
            name for name in reference if name.endswith("embed_tokens.weight")
        )
        source["lm_head.weight"] = reference[embed_name].copy()
    return source


def _runtime_native_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in reference.items()}


def _mixtral_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        if ".switch_mlp.gate_proj.weight" in name:
            prefix = name.replace(".switch_mlp.gate_proj.weight", "")
            for index, chunk in enumerate(value):
                source[f"{prefix}.experts.{index}.w1.weight"] = chunk.copy()
            continue
        if ".switch_mlp.down_proj.weight" in name:
            prefix = name.replace(".switch_mlp.down_proj.weight", "")
            for index, chunk in enumerate(value):
                source[f"{prefix}.experts.{index}.w2.weight"] = chunk.copy()
            continue
        if ".switch_mlp.up_proj.weight" in name:
            prefix = name.replace(".switch_mlp.up_proj.weight", "")
            for index, chunk in enumerate(value):
                source[f"{prefix}.experts.{index}.w3.weight"] = chunk.copy()
            continue
        source[name] = value.copy()
    return source


def _named_experts_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        if ".switch_mlp." in name:
            prefix, projection = name.split(".switch_mlp.", 1)
            for index, chunk in enumerate(value):
                source[f"{prefix}.experts.{index}.{projection}"] = chunk.copy()
            continue
        source[name] = value.copy()
    return source


def _granitemoe_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        if ".block_sparse_moe.switch_mlp.gate_proj.weight" in name:
            prefix = name.replace(".block_sparse_moe.switch_mlp.gate_proj.weight", "")
            up = reference[f"{prefix}.block_sparse_moe.switch_mlp.up_proj.weight"]
            source[f"{prefix}.block_sparse_moe.input_linear.weight"] = np.concatenate(
                [value, up],
                axis=1,
            )
            continue
        if ".block_sparse_moe.switch_mlp.up_proj.weight" in name:
            continue
        if ".block_sparse_moe.switch_mlp.down_proj.weight" in name:
            prefix = name.replace(".block_sparse_moe.switch_mlp.down_proj.weight", "")
            source[f"{prefix}.block_sparse_moe.output_linear.weight"] = value.copy()
            continue
        source[name] = value.copy()
    if "lm_head.weight" not in source:
        source["lm_head.weight"] = reference["model.embed_tokens.weight"].copy()
    return source


def _qwen_vl_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        source_name = name
        source_value = value.copy()
        if name.startswith("language_model."):
            source_name = name[len("language_model.") :]
        elif name.startswith("vision_tower.patch_embed.proj.weight"):
            source_name = "visual.patch_embed.proj.weight"
            source_value = np.transpose(value, (0, 4, 1, 2, 3))
        elif name.startswith("vision_tower."):
            source_name = "visual." + name[len("vision_tower.") :]
        source[source_name] = source_value
    return source


def _pixtral_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        source_name = name
        source_value = value.copy()
        if name.startswith("language_model."):
            source_name = name[len("language_model.") :]
        elif name.startswith("multi_modal_projector."):
            source_name = "model.vision_projection." + name[len("multi_modal_projector.") :]
        elif name.startswith("vision_tower.patch_conv.weight"):
            source_value = np.transpose(value, (0, 3, 1, 2))
        elif name.startswith("vision_tower."):
            source_name = "model.vision_encoder." + name[len("vision_tower.") :]
        source[source_name] = source_value
    return source


def _conv_axis_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        if name.endswith("conv1d.weight") or name.endswith("conv_1d.weight"):
            source[name] = np.moveaxis(value, 1, 2)
        else:
            source[name] = value.copy()
    return source


def _jamba_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        if ".feed_forward.switch_mlp." in name:
            prefix, projection = name.split(".feed_forward.switch_mlp.", 1)
            for index, chunk in enumerate(value):
                source[f"{prefix}.feed_forward.experts.{index}.{projection}"] = chunk.copy()
            continue
        if name.endswith("conv1d.weight"):
            source[name] = np.moveaxis(value, 1, 2)
            continue
        source[name] = value.copy()
    return source


_QWEN35_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _shift_qwen35_norm(name: str, value: np.ndarray) -> np.ndarray:
    if any(name.endswith(suffix) for suffix in _QWEN35_NORM_SUFFIXES) and value.ndim == 1:
        return value - np.array(1.0, dtype=value.dtype)
    return value


def _qwen35_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        source_value = value.copy()
        if name.endswith("conv1d.weight"):
            source_value = np.moveaxis(source_value, 1, 2)
        source[name] = _shift_qwen35_norm(name, source_value)
    return source


def _qwen35_moe_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    consumed_up: set[str] = set()
    for name, value in reference.items():
        if name in consumed_up:
            continue
        if ".switch_mlp.gate_proj.weight" in name:
            prefix = name.replace(".switch_mlp.gate_proj.weight", "")
            up_name = f"{prefix}.switch_mlp.up_proj.weight"
            source[f"{prefix}.experts.gate_up_proj.weight"] = np.concatenate(
                [value, reference[up_name]],
                axis=1,
            )
            consumed_up.add(up_name)
            continue
        if ".switch_mlp.up_proj.weight" in name:
            continue
        if ".switch_mlp.down_proj.weight" in name:
            prefix = name.replace(".switch_mlp.down_proj.weight", "")
            source[f"{prefix}.experts.down_proj.weight"] = value.copy()
            continue
        source_value = value.copy()
        if name.endswith("conv1d.weight"):
            source_value = np.moveaxis(source_value, 1, 2)
        source[name] = _shift_qwen35_norm(name, source_value)
    return source


def _qwen35_vl_source(
    reference: dict[str, np.ndarray],
    _config: dict[str, Any],
) -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {}
    for name, value in reference.items():
        source_name = name
        source_value = value.copy()
        if name.startswith("vision_tower.patch_embed.proj.weight"):
            source_name = "visual.patch_embed.proj.weight"
            source_value = np.transpose(source_value, (0, 4, 1, 2, 3))
        elif name.startswith("vision_tower."):
            source_name = "visual." + name[len("vision_tower.") :]
        elif name.startswith("language_model."):
            if name.endswith("conv1d.weight"):
                source_value = np.moveaxis(source_value, 1, 2)
            source_value = _shift_qwen35_norm(name, source_value)
        source[source_name] = source_value
    return source


def _compare_tensors(
    reference: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
) -> None:
    assert set(reference) == set(actual)
    for name, value in reference.items():
        assert np.array_equal(value, actual[name]), name


def _compare_tensors_allclose(
    reference: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    atol: float,
) -> None:
    assert set(reference) == set(actual)
    for name, value in reference.items():
        assert np.allclose(value, actual[name], atol=atol, rtol=0.0), name


SUPPORTED_CASES = (
    RuntimeCase(
        name="llama",
        runtime_target="llama",
        macro_template="decoder_dense",
        config={
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "tie_word_embeddings": True,
            "max_position_embeddings": 32,
        },
        source_builder=_dense_source_with_extras,
    ),
    RuntimeCase(
        name="gemma",
        runtime_target="gemma",
        macro_template="decoder_dense",
        config={
            "model_type": "gemma",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
        },
        source_builder=_dense_source_with_extras,
    ),
    RuntimeCase(
        name="qwen2",
        runtime_target="qwen2",
        macro_template="decoder_dense",
        config={
            "model_type": "qwen2",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "tie_word_embeddings": False,
            "max_position_embeddings": 32,
        },
        source_builder=_dense_source_with_extras,
    ),
    RuntimeCase(
        name="qwen3",
        runtime_target="qwen3",
        macro_template="decoder_dense",
        config={
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "tie_word_embeddings": False,
            "max_position_embeddings": 32,
        },
        source_builder=_dense_source_with_extras,
    ),
    RuntimeCase(
        name="phi3",
        runtime_target="phi3",
        macro_template="decoder_dense",
        config={
            "model_type": "phi3",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "tie_word_embeddings": False,
            "max_position_embeddings": 32,
        },
        source_builder=_runtime_native_source,
    ),
    RuntimeCase(
        name="mixtral",
        runtime_target="mixtral",
        macro_template="decoder_moe",
        config={
            "model_type": "mixtral",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "vocab_size": 32,
            "tie_word_embeddings": False,
        },
        source_builder=_mixtral_source,
    ),
    RuntimeCase(
        name="qwen2_moe",
        runtime_target="qwen2_moe",
        macro_template="decoder_moe",
        config={
            "model_type": "qwen2_moe",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_experts_per_tok": 1,
            "num_experts": 2,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 8,
            "vocab_size": 32,
            "num_key_value_heads": 1,
            "tie_word_embeddings": False,
        },
        source_builder=_named_experts_source,
    ),
    RuntimeCase(
        name="qwen3_moe",
        runtime_target="qwen3_moe",
        macro_template="decoder_moe",
        config={
            "model_type": "qwen3_moe",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_experts_per_tok": 1,
            "num_experts": 2,
            "moe_intermediate_size": 4,
            "vocab_size": 32,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "tie_word_embeddings": False,
            "max_position_embeddings": 32,
        },
        source_builder=_named_experts_source,
    ),
    RuntimeCase(
        name="granitemoe",
        runtime_target="granitemoe",
        macro_template="decoder_moe",
        config={
            "model_type": "granitemoe",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "tie_word_embeddings": True,
        },
        source_builder=_granitemoe_source,
    ),
    RuntimeCase(
        name="qwen2_vl",
        runtime_target="qwen2_vl",
        macro_template="multimodal_decoder",
        config={
            "model_type": "qwen2_vl",
            "text_config": {
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "tie_word_embeddings": False,
                "max_position_embeddings": 32,
            },
            "vision_config": {
                "depth": 1,
                "embed_dim": 8,
                "hidden_size": 8,
                "num_heads": 2,
                "image_size": 4,
                "patch_size": 2,
                "in_channels": 3,
                "mlp_ratio": 2.0,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
            "image_token_id": 31,
            "video_token_id": 30,
        },
        source_builder=_qwen_vl_source,
        multimodal=True,
    ),
    RuntimeCase(
        name="qwen3_vl",
        runtime_target="qwen3_vl",
        macro_template="multimodal_decoder",
        config={
            "model_type": "qwen3_vl",
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "tie_word_embeddings": False,
                "max_position_embeddings": 32,
            },
            "vision_config": {
                "depth": 1,
                "embed_dim": 8,
                "hidden_size": 8,
                "num_heads": 2,
                "image_size": 4,
                "patch_size": 2,
                "in_channels": 3,
                "mlp_ratio": 2.0,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
            "image_token_id": 31,
            "video_token_id": 30,
        },
        source_builder=_qwen_vl_source,
        multimodal=True,
    ),
    RuntimeCase(
        name="pixtral",
        runtime_target="pixtral",
        macro_template="multimodal_decoder",
        config={
            "model_type": "pixtral",
            "text_config": {
                "model_type": "llama",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "tie_word_embeddings": False,
                "max_position_embeddings": 32,
            },
            "vision_config": {
                "model_type": "pixtral",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "head_dim": 4,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "image_size": 4,
                "patch_size": 2,
                "projection_dim": 8,
                "num_channels": 3,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            },
            "image_token_id": 31,
        },
        source_builder=_pixtral_source,
        multimodal=True,
    ),
    RuntimeCase(
        name="lfm2_vl",
        runtime_target="lfm2_vl",
        macro_template="multimodal_decoder",
        config={
            "model_type": "lfm2_vl",
            "text_config": {
                "model_type": "lfm2",
                "vocab_size": 32,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "conv_L_cache": 4,
                "block_dim": 8,
                "block_ff_dim": 16,
                "full_attn_idxs": [0],
            },
            "vision_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 4,
                "patch_size": 2,
                "num_patches": 4,
            },
            "image_token_id": 31,
        },
        source_builder=_runtime_native_source,
        multimodal=True,
    ),
    RuntimeCase(
        name="mamba2",
        runtime_target="mamba2",
        macro_template="ssm_hybrid",
        config={
            "model_type": "mamba2",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "vocab_size": 32,
            "num_heads": 2,
            "head_dim": 4,
            "state_size": 4,
            "conv_kernel": 3,
            "n_groups": 1,
            "tie_word_embeddings": True,
        },
        source_builder=_conv_axis_source,
    ),
    RuntimeCase(
        name="mamba",
        runtime_target="mamba",
        macro_template="ssm_hybrid",
        config={
            "model_type": "mamba",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "state_size": 4,
            "num_hidden_layers": 1,
            "conv_kernel": 3,
            "tie_word_embeddings": True,
        },
        source_builder=_conv_axis_source,
    ),
    RuntimeCase(
        name="recurrent_gemma",
        runtime_target="recurrent_gemma",
        macro_template="ssm_hybrid",
        config={
            "model_type": "recurrent_gemma",
            "attention_bias": True,
            "conv1d_width": 3,
            "hidden_size": 8,
            "intermediate_size": 16,
            "logits_soft_cap": 30.0,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_window_size": 8,
            "vocab_size": 32,
            "block_types": ["recurrent", "attention"],
        },
        source_builder=_conv_axis_source,
    ),
    RuntimeCase(
        name="jamba",
        runtime_target="jamba",
        macro_template="ssm_hybrid",
        config={
            "model_type": "jamba",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "attn_layer_offset": 0,
            "attn_layer_period": 2,
            "expert_layer_offset": 0,
            "expert_layer_period": 1,
            "mamba_d_conv": 3,
            "mamba_d_state": 4,
            "mamba_expand": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "vocab_size": 32,
            "tie_word_embeddings": False,
        },
        source_builder=_jamba_source,
    ),
    RuntimeCase(
        name="qwen3_5",
        runtime_target="qwen3_5",
        macro_template="ssm_hybrid",
        config={
            "model_type": "qwen3_5",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 3,
            "tie_word_embeddings": False,
            "head_dim": 8,
            "full_attention_interval": 2,
            "max_position_embeddings": 32,
            "rope_parameters": {
                "type": "default",
                "rope_theta": 100000.0,
                "partial_rotary_factor": 0.25,
            },
        },
        source_builder=_qwen35_source,
        tensor_compare="allclose",
        tensor_atol=1e-6,
    ),
    RuntimeCase(
        name="qwen3_5_moe",
        runtime_target="qwen3_5_moe",
        macro_template="ssm_hybrid",
        config={
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 3,
                "tie_word_embeddings": False,
                "head_dim": 8,
                "full_attention_interval": 2,
                "max_position_embeddings": 32,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 32,
                "moe_intermediate_size": 16,
                "norm_topk_prob": True,
                "mlp_only_layers": [],
                "rope_parameters": {
                    "type": "default",
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        },
        source_builder=_qwen35_moe_source,
        tensor_compare="allclose",
        tensor_atol=1e-6,
    ),
    RuntimeCase(
        name="qwen3_5_vl",
        runtime_target="qwen3_5_vl",
        macro_template="multimodal_decoder",
        config={
            "model_type": "qwen3_5_vl",
            "text_config": {
                "model_type": "qwen3_5",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 3,
                "tie_word_embeddings": False,
                "head_dim": 8,
                "full_attention_interval": 2,
                "max_position_embeddings": 32,
                "rope_parameters": {
                    "type": "default",
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
            "vision_config": {
                "model_type": "qwen3_5_vl",
                "depth": 1,
                "hidden_size": 8,
                "out_hidden_size": 16,
                "num_heads": 2,
                "patch_size": 2,
                "in_channels": 3,
                "mlp_ratio": 2.0,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
            "image_token_id": 31,
            "video_token_id": 30,
        },
        source_builder=_qwen35_vl_source,
        multimodal=True,
        tensor_compare="allclose",
        tensor_atol=1e-6,
    ),
)


@pytest.mark.parametrize("case", SUPPORTED_CASES, ids=[case.name for case in SUPPORTED_CASES])
def test_convert_source_e2e_runtime(case: RuntimeCase, tmp_path: Path) -> None:
    reference = _build_reference_tensors(
        case.runtime_target,
        case.config,
        multimodal=case.multimodal,
    )
    source_tensors = case.source_builder(reference, case.config)
    source_dir = tmp_path / case.name / "source"
    output_dir = tmp_path / case.name / "output"
    _save_fixture(source_dir, case.config, source_tensors)

    result = convert_source(
        source_dir,
        output_dir,
        options=ConversionOptions(verification_mode=VerificationMode.REQUIRED),
    )

    assert result.canonical_ir.identity.macro_template.value == case.macro_template
    assert result.plan.runtime_target_model_type == case.runtime_target

    converted = load_file(str(output_dir / "model.safetensors"))
    if case.tensor_compare == "allclose":
        _compare_tensors_allclose(reference, converted, atol=case.tensor_atol)
    else:
        _compare_tensors(reference, converted)

    verification = verify_output(
        output_dir,
        options=ConversionOptions(verification_mode=VerificationMode.REQUIRED),
    )
    assert verification.status.value == "passed"

    model = load_model(output_dir, lazy=False)
    logits = model(mx.array([[1, 2]], dtype=mx.int32))
    assert tuple(int(dim) for dim in logits.shape) == (1, 2, 32)

    tokenizer = load_tokenizer(output_dir)
    assert tokenizer.vocab_size == 5

    loaded_model, loaded_tokenizer = load_model_and_tokenizer(
        output_dir,
        ModelConfig(model_path=str(output_dir), preload=True, lazy_load=False),
    )
    assert loaded_model is not None
    assert loaded_tokenizer.vocab_size == 5


@pytest.mark.parametrize(
    ("name", "config", "tensors", "expected_macro"),
    [
        (
            "encoder_decoder_t5",
            {
                "model_type": "t5",
                "hidden_size": 8,
                "is_encoder_decoder": True,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "vocab_size": 32,
            },
            {
                "encoder.block.0.weight": np.ones((2, 2), dtype=np.float32),
                "decoder.block.0.weight": np.ones((2, 2), dtype=np.float32),
            },
            "encoder_decoder",
        ),
        (
            "multimodal_encoder_decoder_fixture",
            {
                "model_type": "mm_enc_dec",
                "hidden_size": 8,
                "is_encoder_decoder": True,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "vocab_size": 32,
                "vision_config": {"hidden_size": 8},
            },
            {
                "encoder.block.0.weight": np.ones((2, 2), dtype=np.float32),
                "decoder.block.0.weight": np.ones((2, 2), dtype=np.float32),
                "vision_tower.patch_embed.proj.weight": np.ones(
                    (2, 1, 1, 1, 1),
                    dtype=np.float32,
                ),
            },
            "multimodal_encoder_decoder",
        ),
    ],
)
def test_convert_source_hard_fails_for_unsupported_runtime_targets(
    name: str,
    config: dict[str, Any],
    tensors: dict[str, np.ndarray],
    expected_macro: str,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / name / "source"
    output_dir = tmp_path / name / "output"
    _save_fixture(source_dir, config, tensors, tokenizer=False)

    with pytest.raises(UnsupportedRuntimeTargetError):
        convert_source(
            source_dir,
            output_dir,
            options=ConversionOptions(verification_mode=VerificationMode.REQUIRED),
        )

    manifest = json.loads((output_dir / "conversion_manifest.json").read_text())
    assert manifest["macro_template"] == expected_macro
    assert manifest["failure_phase"] == "planning"


def test_verify_output_blocks_real_shape_mismatch(tmp_path: Path) -> None:
    case = next(case for case in SUPPORTED_CASES if case.name == "qwen3")
    reference = _build_reference_tensors(
        case.runtime_target,
        case.config,
        multimodal=case.multimodal,
    )
    source_tensors = case.source_builder(reference, case.config)
    source_dir = tmp_path / "qwen3" / "source"
    output_dir = tmp_path / "qwen3" / "output"
    _save_fixture(source_dir, case.config, source_tensors)

    convert_source(
        source_dir,
        output_dir,
        options=ConversionOptions(verification_mode=VerificationMode.REQUIRED),
    )

    converted = load_file(str(output_dir / "model.safetensors"))
    corrupted_name = "model.layers.0.self_attn.k_proj.weight"
    converted[corrupted_name] = converted[corrupted_name].T.copy()
    save_file(converted, str(output_dir / "model.safetensors"))

    with pytest.raises(ConversionVerificationError):
        verify_output(
            output_dir,
            options=ConversionOptions(verification_mode=VerificationMode.REQUIRED),
        )
