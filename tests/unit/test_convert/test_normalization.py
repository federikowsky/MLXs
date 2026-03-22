from __future__ import annotations

from pathlib import Path

import pytest

from mlxs.convert.normalization import (
    _compatibility_macro_template_from_traits,
    normalize_inspection,
)
from mlxs.convert.types import (
    ArchitectureTraits,
    ConversionOptions,
    ExpertLayoutKind,
    InspectionReport,
    MacroTemplate,
    Modality,
    SequenceFamilyKind,
    SourceKind,
    TensorInfo,
    TopologyKind,
)


def _inspection(config: dict[str, object], tensor_names: list[str]) -> InspectionReport:
    return InspectionReport(
        source_kind=SourceKind.LOCAL,
        source_id="fixture",
        resolved_path=Path("/tmp/model"),
        config_path=Path("/tmp/model/config.json"),
        config=config,
        weight_format="safetensors",
        sharded=False,
        shard_files=("model.safetensors",),
        tensor_infos=tuple(
            TensorInfo(name=name, shape=(4, 4), dtype="F32", file="model.safetensors")
            for name in tensor_names
        ),
        tokenizer_artifacts=("tokenizer.json",),
        multimodal_artifacts=(),
        custom_code_indicators=(),
    )


def test_normalize_dense_decoder() -> None:
    inspection = _inspection(
        {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 256,
            "tie_word_embeddings": False,
        },
        ["model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight"],
    )

    ir = normalize_inspection(inspection, options=ConversionOptions())

    assert ir.identity.macro_template.value == "decoder_dense"
    assert ir.identity.runtime_target_model_type == "qwen3"
    assert ir.traits.modality == Modality.TEXT
    assert ir.traits.topology_kind == TopologyKind.DECODER
    assert ir.traits.expert_layout == ExpertLayoutKind.DENSE
    assert ir.traits.sequence_family == SequenceFamilyKind.ATTENTION
    assert ir.topology.attention_variant == "gqa"


def test_normalize_decoder_moe() -> None:
    inspection = _inspection(
        {
            "model_type": "mixtral",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 256,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
        [
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
        ],
    )

    ir = normalize_inspection(inspection)

    assert ir.identity.macro_template.value == "decoder_moe"
    assert ir.traits.expert_layout == ExpertLayoutKind.MOE
    assert ir.traits.sequence_family == SequenceFamilyKind.ATTENTION
    assert ir.topology.density.value == "moe"
    assert ir.config.values["num_experts"] == 8


def test_normalize_multimodal_decoder() -> None:
    inspection = _inspection(
        {
            "model_type": "qwen2",
            "text_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 256,
            },
            "vision_config": {"hidden_size": 64},
            "image_token_id": 151655,
        },
        [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "visual.patch_embed.proj.weight",
        ],
    )

    ir = normalize_inspection(inspection)

    assert ir.identity.macro_template.value == "multimodal_decoder"
    assert ir.topology.multimodal is True
    assert ir.traits.modality == Modality.MULTIMODAL
    assert ir.traits.topology_kind == TopologyKind.DECODER
    assert ir.identity.runtime_target_model_type == "qwen2"
    assert ir.conversion.canonical_output_config["model_type"] == "qwen2"
    assert "visual->vision_tower" in ir.tensor_layout.naming_aliases_discovered


def test_normalize_kimi_vl_multimodal_decoder() -> None:
    inspection = _inspection(
        {
            "model_type": "kimi_vl",
            "text_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 256,
                "qk_rope_head_dim": 16,
                "qk_nope_head_dim": 16,
                "v_head_dim": 32,
            },
            "vision_config": {
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
            },
            "media_placeholder_token_id": 151655,
        },
        [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "vision_tower.patch_embed.proj.weight",
            "multi_modal_projector.linear_1.weight",
        ],
    )

    ir = normalize_inspection(inspection)

    assert ir.identity.macro_template.value == "multimodal_decoder"
    assert ir.identity.runtime_target_model_type == "kimi_vl"
    assert ir.identity.supported_by_runtime is True
    assert ir.traits.modality == Modality.MULTIMODAL
    assert ir.topology.multimodal is True


@pytest.mark.parametrize(
    "legacy_model_type",
    ["qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5_vl", "lfm2_vl"],
)
def test_normalize_removed_legacy_identifiers_as_unsupported_runtime_targets(
    legacy_model_type: str,
) -> None:
    inspection = _inspection(
        {
            "model_type": legacy_model_type,
            "text_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
                "vocab_size": 256,
            },
            "vision_config": {"hidden_size": 64},
            "image_token_id": 151655,
        },
        [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "visual.patch_embed.proj.weight",
        ],
    )

    ir = normalize_inspection(inspection)

    assert ir.identity.runtime_target_model_type == legacy_model_type
    assert ir.identity.supported_by_runtime is False


def test_normalize_qwen3_5_moe_multimodal_preserves_hybrid_moe_traits() -> None:
    inspection = _inspection(
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
                "vocab_size": 256,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 2,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64,
                "moe_intermediate_size": 64,
            },
            "vision_config": {"hidden_size": 64},
            "image_token_id": 151655,
        },
        [
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "vision_tower.patch_embed.proj.weight",
        ],
    )

    ir = normalize_inspection(inspection)

    assert ir.identity.macro_template.value == "multimodal_decoder"
    assert ir.identity.runtime_target_model_type == "qwen3_5_moe"
    assert ir.identity.supported_by_runtime is True
    assert ir.traits.modality == Modality.MULTIMODAL
    assert ir.traits.topology_kind == TopologyKind.DECODER
    assert ir.traits.expert_layout == ExpertLayoutKind.MOE
    assert ir.traits.sequence_family == SequenceFamilyKind.SSM_HYBRID
    assert ir.topology.multimodal is True
    assert ir.topology.ssm_hybrid is True
    assert ir.topology.density.value == "moe"


def test_normalize_ssm_hybrid() -> None:
    inspection = _inspection(
        {
            "model_type": "mamba2",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "vocab_size": 256,
            "state_size": 16,
            "conv_kernel": 4,
        },
        ["backbone.layers.0.mixer.conv1d.weight", "backbone.layers.0.mixer.A_log"],
    )

    ir = normalize_inspection(inspection)

    assert ir.identity.macro_template.value == "ssm_hybrid"
    assert ir.traits.expert_layout == ExpertLayoutKind.DENSE
    assert ir.traits.sequence_family == SequenceFamilyKind.SSM_HYBRID
    assert ir.topology.ssm_hybrid is True
    assert "conv_axis_sensitive" in ir.evidence.selected_rules


def test_normalize_encoder_decoder_marks_unsupported_runtime() -> None:
    inspection = _inspection(
        {
            "model_type": "t5",
            "hidden_size": 64,
            "is_encoder_decoder": True,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "vocab_size": 256,
        },
        ["encoder.block.0.weight", "decoder.block.0.weight"],
    )

    ir = normalize_inspection(inspection)

    assert ir.identity.macro_template.value == "encoder_decoder"
    assert ir.traits.topology_kind == TopologyKind.ENCODER_DECODER
    assert ir.identity.supported_by_runtime is False


@pytest.mark.parametrize(
    ("traits", "expected"),
    [
        (
            ArchitectureTraits(
                modality=Modality.TEXT,
                topology_kind=TopologyKind.DECODER,
                expert_layout=ExpertLayoutKind.DENSE,
                sequence_family=SequenceFamilyKind.ATTENTION,
            ),
            MacroTemplate.DECODER_DENSE,
        ),
        (
            ArchitectureTraits(
                modality=Modality.TEXT,
                topology_kind=TopologyKind.DECODER,
                expert_layout=ExpertLayoutKind.MOE,
                sequence_family=SequenceFamilyKind.ATTENTION,
            ),
            MacroTemplate.DECODER_MOE,
        ),
        (
            ArchitectureTraits(
                modality=Modality.TEXT,
                topology_kind=TopologyKind.DECODER,
                expert_layout=ExpertLayoutKind.MOE,
                sequence_family=SequenceFamilyKind.SSM_HYBRID,
            ),
            MacroTemplate.SSM_HYBRID,
        ),
        (
            ArchitectureTraits(
                modality=Modality.MULTIMODAL,
                topology_kind=TopologyKind.DECODER,
                expert_layout=ExpertLayoutKind.MOE,
                sequence_family=SequenceFamilyKind.SSM_HYBRID,
            ),
            MacroTemplate.MULTIMODAL_DECODER,
        ),
        (
            ArchitectureTraits(
                modality=Modality.MULTIMODAL,
                topology_kind=TopologyKind.ENCODER_DECODER,
                expert_layout=ExpertLayoutKind.DENSE,
                sequence_family=SequenceFamilyKind.ATTENTION,
            ),
            MacroTemplate.MULTIMODAL_ENCODER_DECODER,
        ),
    ],
)
def test_compatibility_macro_template_is_derived_from_traits(
    traits: ArchitectureTraits,
    expected: MacroTemplate,
) -> None:
    assert _compatibility_macro_template_from_traits(traits) == expected
