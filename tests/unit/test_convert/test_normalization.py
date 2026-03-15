from __future__ import annotations

from pathlib import Path

from mlxs.convert.normalization import normalize_inspection
from mlxs.convert.types import ConversionOptions, InspectionReport, SourceKind, TensorInfo


def _inspection(config: dict, tensor_names: list[str]) -> InspectionReport:
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
    assert ir.topology.density.value == "moe"
    assert ir.config.values["num_experts"] == 8


def test_normalize_multimodal_decoder() -> None:
    inspection = _inspection(
        {
            "model_type": "qwen2_vl",
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
    assert "visual->vision_tower" in ir.tensor_layout.naming_aliases_discovered


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
    assert ir.identity.supported_by_runtime is False
