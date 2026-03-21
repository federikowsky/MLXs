from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mlxs.convert.errors import MissingRequiredTensorError, UnsupportedRuntimeTargetError
from mlxs.convert.planning import _alias_candidates, build_conversion_plan
from mlxs.convert.types import (
    ArchitectureTraits,
    CanonicalIR,
    ConversionOptions,
    DensityKind,
    ExpertLayoutKind,
    InspectionReport,
    IRAmbiguities,
    IRConfig,
    IRConversion,
    IREvidence,
    IRIdentity,
    IRSource,
    IRTensorLayout,
    IRTokenizer,
    IRTopology,
    MacroTemplate,
    Modality,
    RuntimeTensorSchemaEntry,
    SequenceFamilyKind,
    SourceKind,
    TensorInfo,
    TopologyKind,
)


def _inspection(tensors: list[str]) -> InspectionReport:
    return InspectionReport(
        source_kind=SourceKind.LOCAL,
        source_id="fixture",
        resolved_path=Path("/tmp/model"),
        config_path=Path("/tmp/model/config.json"),
        config={"model_type": "mixtral", "num_local_experts": 2},
        weight_format="safetensors",
        sharded=False,
        shard_files=("model.safetensors",),
        tensor_infos=tuple(
            TensorInfo(name=name, shape=(2, 4), dtype="F32", file="model.safetensors")
            for name in tensors
        ),
        tokenizer_artifacts=("tokenizer.json",),
        multimodal_artifacts=(),
        custom_code_indicators=(),
    )


def _canonical_ir(*, supported: bool = True) -> CanonicalIR:
    return CanonicalIR(
        source=IRSource(
            kind=SourceKind.LOCAL,
            source_id="fixture",
            resolved_path=Path("/tmp/model"),
            weight_format="safetensors",
            sharded=False,
            shard_list=("model.safetensors",),
            source_config_artifacts_present=("config.json",),
            tokenizer_artifacts_present=("tokenizer.json",),
            multimodal_artifacts_present=(),
            custom_code_indicators=(),
        ),
        identity=IRIdentity(
            macro_template=MacroTemplate.DECODER_MOE,
            modality=Modality.TEXT,
            architecture_label="mixtral",
            variant_label=None,
            runtime_target_model_type="mixtral",
            supported_by_runtime=supported,
        ),
        traits=ArchitectureTraits(
            modality=Modality.TEXT,
            topology_kind=TopologyKind.DECODER,
            expert_layout=ExpertLayoutKind.MOE,
            sequence_family=SequenceFamilyKind.ATTENTION,
        ),
        topology=IRTopology(
            backbone_type="decoder",
            decoder_only=True,
            encoder_decoder=False,
            multimodal=False,
            ssm_hybrid=False,
            density=DensityKind.MOE,
            attention_presence=True,
            attention_variant="gqa",
            qkv_layout="split_qkv",
            mlp_variant="switch_glu",
            norm_type="rmsnorm",
            rope_variant="rope",
            embedding_tying=False,
            projector_presence=False,
            tower_presence=False,
            recurrent_traits=(),
        ),
        config=IRConfig(
            values={"num_experts": 2},
            raw_config={"model_type": "mixtral", "num_local_experts": 2},
            required_fields=("hidden_size",),
        ),
        tensor_layout=IRTensorLayout(
            selected_source_tensor_profile="moe_triplets",
            canonical_internal_tensor_profile="runtime_post_sanitize",
            tensor_groups_present=("core", "moe"),
            optional_tensor_groups_present=("moe",),
            missing_required_groups=(),
            fused_split_markers=("moe_triplets",),
            naming_aliases_discovered=(),
        ),
        tokenizer=IRTokenizer(
            tokenizer_type="tokenizer_json",
            artifact_paths=("tokenizer.json",),
            special_tokens={},
            vocab_metadata={"vocab_size": 256},
            chat_template_present=False,
        ),
        conversion=IRConversion(
            dtype_policy="preserve",
            target_format_version="mlxs-converter-v1",
            quantization_mode=None,
            shard_output_policy="single",
            cast_rules=("preserve",),
            save_options={},
            canonical_output_config={"model_type": "mixtral"},
        ),
        evidence=IREvidence(
            matched_config_keys=("model_type",),
            matched_tensor_patterns=("moe_triplets",),
            matched_shape_traits=(),
            matched_tokenizer_traits=("tokenizer.json",),
            selected_rules=("moe_triplets",),
            model_assisted_normalization_usage=None,
        ),
        ambiguities=IRAmbiguities(
            ambiguity_flags=(),
            selected_resolution_path="deterministic_rules",
            confidence_score=None,
            fallback_markers=(),
        ),
    )


def _family_ir(
    runtime_target_model_type: str,
    *,
    multimodal: bool,
) -> CanonicalIR:
    base = _canonical_ir()
    return replace(
        base,
        identity=replace(
            base.identity,
            macro_template=(
                MacroTemplate.MULTIMODAL_DECODER
                if multimodal
                else MacroTemplate.DECODER_DENSE
            ),
            modality=Modality.MULTIMODAL if multimodal else Modality.TEXT,
            architecture_label=runtime_target_model_type,
            runtime_target_model_type=runtime_target_model_type,
            supported_by_runtime=True,
        ),
        traits=replace(
            base.traits,
            modality=Modality.MULTIMODAL if multimodal else Modality.TEXT,
            topology_kind=TopologyKind.DECODER,
            expert_layout=ExpertLayoutKind.DENSE,
            sequence_family=SequenceFamilyKind.ATTENTION,
        ),
        topology=replace(
            base.topology,
            backbone_type="multimodal_decoder" if multimodal else "decoder",
            multimodal=multimodal,
            density=DensityKind.DENSE,
            projector_presence=multimodal,
            tower_presence=multimodal,
        ),
        config=replace(
            base.config,
            values={},
            raw_config={"model_type": runtime_target_model_type},
        ),
        tensor_layout=replace(
            base.tensor_layout,
            selected_source_tensor_profile=(
                "visual_prefix" if multimodal else "runtime_native"
            ),
            tensor_groups_present=("core", "multimodal") if multimodal else ("core",),
            optional_tensor_groups_present=("multimodal",) if multimodal else (),
            fused_split_markers=("visual_prefix",) if multimodal else (),
            naming_aliases_discovered=(
                ("language_model_prefix", "visual_prefix")
                if multimodal
                else ("language_model_prefix",)
            ),
        ),
        conversion=replace(
            base.conversion,
            canonical_output_config={"model_type": runtime_target_model_type},
        ),
        evidence=replace(
            base.evidence,
            matched_config_keys=("model_type",),
            matched_tensor_patterns=(
                ("visual_prefix",) if multimodal else ("language_model_prefix",)
            ),
            selected_rules=(runtime_target_model_type,),
        ),
    )


def test_build_conversion_plan_stacks_triplet_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    inspection = _inspection(
        [
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
        ]
    )
    ir = _canonical_ir()

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [
            RuntimeTensorSchemaEntry(
                name="model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight",
                shape=(2, 2, 4),
            )
        ],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    assert len(plan.mappings) == 1
    mapping = plan.mappings[0]
    assert mapping.target_name.endswith("switch_mlp.gate_proj.weight")
    assert mapping.transforms[0].kind.value == "stack"
    assert mapping.rule_id == "stack_triplet_experts"
    assert mapping.match_layer == "structural"
    assert mapping.adapter_name is None


def test_build_conversion_plan_rejects_unsupported_runtime() -> None:
    inspection = _inspection([])
    ir = _canonical_ir(supported=False)

    with pytest.raises(UnsupportedRuntimeTargetError):
        build_conversion_plan(inspection, ir)


def test_alias_candidates_are_deterministic_and_keep_exact_target_first() -> None:
    target_name = "language_model.model.layers.0.mlp.fc1.weight"

    first = _alias_candidates(target_name)
    second = _alias_candidates(target_name)

    assert first == second
    assert first[0] == target_name


def test_build_conversion_plan_prefers_exact_match_over_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_name = "model.embed_tokens.weight"
    inspection = _inspection(
        [
            target_name,
            "language_model.model.embed_tokens.weight",
        ]
    )
    ir = _canonical_ir()

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [RuntimeTensorSchemaEntry(name=target_name, shape=(2, 4))],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    assert plan.mappings[0].source_names == (target_name,)
    assert plan.mappings[0].rule_id == "exact"
    assert plan.mappings[0].match_layer == "exact"
    assert plan.mappings[0].adapter_name is None


def test_build_conversion_plan_uses_qwen_family_adapter_for_language_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["language_model.model.embed_tokens.weight"])
    ir = _family_ir("qwen3", multimodal=False)

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [RuntimeTensorSchemaEntry(name="model.embed_tokens.weight", shape=(2, 4))],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    mapping = plan.mappings[0]
    assert mapping.source_names == ("language_model.model.embed_tokens.weight",)
    assert mapping.rule_id == "qwen_family:language_model_prefix"
    assert mapping.match_layer == "family_adapter"
    assert mapping.adapter_name == "qwen_family"


def test_build_conversion_plan_uses_qwen_family_adapter_for_vision_model_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["vision_model.encoder.weight"])
    ir = _family_ir("qwen2", multimodal=True)

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [RuntimeTensorSchemaEntry(name="vision_tower.encoder.weight", shape=(2, 4))],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    mapping = plan.mappings[0]
    assert mapping.source_names == ("vision_model.encoder.weight",)
    assert mapping.rule_id == "qwen_family:vision_model_prefix"
    assert mapping.match_layer == "family_adapter"
    assert mapping.adapter_name == "qwen_family"


def test_build_conversion_plan_uses_qwen35_adapter_for_language_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["language_model.model.embed_tokens.weight"])
    ir = _family_ir("qwen3_5", multimodal=False)

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [RuntimeTensorSchemaEntry(name="model.embed_tokens.weight", shape=(2, 4))],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    mapping = plan.mappings[0]
    assert mapping.source_names == ("language_model.model.embed_tokens.weight",)
    assert mapping.rule_id == "qwen35_family:language_model_prefix"
    assert mapping.match_layer == "family_adapter"
    assert mapping.adapter_name == "qwen35_family"


def test_build_conversion_plan_uses_qwen35_adapter_for_vision_model_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["vision_model.encoder.weight"])
    ir = _family_ir("qwen3_5", multimodal=True)

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [RuntimeTensorSchemaEntry(name="vision_tower.encoder.weight", shape=(2, 4))],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    mapping = plan.mappings[0]
    assert mapping.source_names == ("vision_model.encoder.weight",)
    assert mapping.rule_id == "qwen35_family:vision_model_prefix"
    assert mapping.match_layer == "family_adapter"
    assert mapping.adapter_name == "qwen35_family"


def test_build_conversion_plan_uses_qwen35_moe_adapter_for_language_wrapper_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["model.embed_tokens.weight"])
    ir = _family_ir("qwen3_5_moe", multimodal=False)

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [
            RuntimeTensorSchemaEntry(
                name="language_model.model.embed_tokens.weight",
                shape=(2, 4),
            )
        ],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    mapping = plan.mappings[0]
    assert mapping.source_names == ("model.embed_tokens.weight",)
    assert mapping.rule_id == "qwen35_moe_family:strip_language_model_prefix"
    assert mapping.match_layer == "family_adapter"
    assert mapping.adapter_name == "qwen35_moe_family"


def test_build_conversion_plan_uses_qwen35_moe_adapter_for_gate_up_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["language_model.model.layers.0.mlp.experts.gate_up_proj.weight"])
    ir = _family_ir("qwen3_5_moe", multimodal=False)

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [
            RuntimeTensorSchemaEntry(
                name="language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                shape=(2, 8, 8),
            )
        ],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    mapping = plan.mappings[0]
    assert mapping.source_names == (
        "language_model.model.layers.0.mlp.experts.gate_up_proj.weight",
    )
    assert mapping.rule_id == "qwen35_moe_family:gate_up_split"
    assert mapping.match_layer == "family_adapter"
    assert mapping.adapter_name == "qwen35_moe_family"
    assert mapping.transforms[0].kind.value == "slice"


def test_build_conversion_plan_keeps_non_pilot_family_on_generic_alias_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["model.vision_encoder.encoder.weight"])
    ir = _family_ir("pixtral", multimodal=True)

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [RuntimeTensorSchemaEntry(name="vision_tower.encoder.weight", shape=(2, 4))],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    mapping = plan.mappings[0]
    assert mapping.source_names == ("model.vision_encoder.encoder.weight",)
    assert mapping.rule_id == "generic_alias"
    assert mapping.match_layer == "generic_alias"
    assert mapping.adapter_name is None


def test_build_conversion_plan_fails_when_required_tensor_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection([])
    ir = _canonical_ir()

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [RuntimeTensorSchemaEntry(name="missing.weight", shape=(2, 2))],
    )

    with pytest.raises(MissingRequiredTensorError):
        build_conversion_plan(inspection, ir)


def test_build_conversion_plan_normalizes_patch_conv_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["vision_tower.patch_conv.weight"])
    ir = CanonicalIR(
        source=IRSource(
            kind=SourceKind.LOCAL,
            source_id="fixture",
            resolved_path=Path("/tmp/model"),
            weight_format="safetensors",
            sharded=False,
            shard_list=("model.safetensors",),
            source_config_artifacts_present=("config.json",),
            tokenizer_artifacts_present=("tokenizer.json",),
            multimodal_artifacts_present=(),
            custom_code_indicators=(),
        ),
        identity=IRIdentity(
            macro_template=MacroTemplate.MULTIMODAL_DECODER,
            modality=Modality.MULTIMODAL,
            architecture_label="pixtral",
            variant_label=None,
            runtime_target_model_type="pixtral",
            supported_by_runtime=True,
        ),
        traits=ArchitectureTraits(
            modality=Modality.MULTIMODAL,
            topology_kind=TopologyKind.DECODER,
            expert_layout=ExpertLayoutKind.DENSE,
            sequence_family=SequenceFamilyKind.ATTENTION,
        ),
        topology=IRTopology(
            backbone_type="multimodal_decoder",
            decoder_only=True,
            encoder_decoder=False,
            multimodal=True,
            ssm_hybrid=False,
            density=DensityKind.DENSE,
            attention_presence=True,
            attention_variant="gqa",
            qkv_layout="split_qkv",
            mlp_variant="swiglu",
            norm_type="rmsnorm",
            rope_variant="rope",
            embedding_tying=False,
            projector_presence=False,
            tower_presence=True,
            recurrent_traits=(),
        ),
        config=IRConfig(
            values={},
            raw_config={"model_type": "pixtral"},
            required_fields=("hidden_size",),
        ),
        tensor_layout=IRTensorLayout(
            selected_source_tensor_profile="visual_prefix",
            canonical_internal_tensor_profile="runtime_post_sanitize",
            tensor_groups_present=("core", "multimodal"),
            optional_tensor_groups_present=("multimodal",),
            missing_required_groups=(),
            fused_split_markers=("visual_prefix",),
            naming_aliases_discovered=("visual->vision_tower",),
        ),
        tokenizer=IRTokenizer(
            tokenizer_type="tokenizer_json",
            artifact_paths=("tokenizer.json",),
            special_tokens={},
            vocab_metadata={"vocab_size": 256},
            chat_template_present=False,
        ),
        conversion=IRConversion(
            dtype_policy="preserve",
            target_format_version="mlxs-converter-v1",
            quantization_mode=None,
            shard_output_policy="single",
            cast_rules=("preserve",),
            save_options={},
            canonical_output_config={"model_type": "pixtral"},
        ),
        evidence=IREvidence(
            matched_config_keys=("model_type",),
            matched_tensor_patterns=("visual_prefix",),
            matched_shape_traits=(),
            matched_tokenizer_traits=("tokenizer.json",),
            selected_rules=("visual_prefix",),
            model_assisted_normalization_usage=None,
        ),
        ambiguities=IRAmbiguities(
            ambiguity_flags=(),
            selected_resolution_path="deterministic_rules",
            confidence_score=None,
            fallback_markers=(),
        ),
    )

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [
            RuntimeTensorSchemaEntry(
                name="vision_tower.patch_conv.weight",
                shape=(8, 2, 2, 3),
            )
        ],
    )

    inspection = InspectionReport(
        source_kind=SourceKind.LOCAL,
        source_id="fixture",
        resolved_path=Path("/tmp/model"),
        config_path=Path("/tmp/model/config.json"),
        config={"model_type": "pixtral"},
        weight_format="safetensors",
        sharded=False,
        shard_files=("model.safetensors",),
        tensor_infos=(
            TensorInfo(
                name="vision_tower.patch_conv.weight",
                shape=(8, 3, 2, 2),
                dtype="F32",
                file="model.safetensors",
            ),
        ),
        tokenizer_artifacts=("tokenizer.json",),
        multimodal_artifacts=(),
        custom_code_indicators=(),
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    assert plan.mappings[0].transforms[0].kind.value == "transpose"
    assert plan.mappings[0].transforms[0].permutation == (0, 2, 3, 1)


def test_build_conversion_plan_stacks_named_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(
        [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
        ]
    )
    ir = _canonical_ir()

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [
            RuntimeTensorSchemaEntry(
                name="model.layers.0.mlp.switch_mlp.gate_proj.weight",
                shape=(2, 2, 4),
            )
        ],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    assert len(plan.mappings) == 1
    mapping = plan.mappings[0]
    assert mapping.transforms[0].kind.value == "stack"
    assert mapping.rule_id == "stack_named_experts"
    assert mapping.match_layer == "structural"
    assert mapping.adapter_name is None


def test_build_conversion_plan_splits_switch_mlp_input_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = _inspection(["model.layers.0.block_sparse_moe.input_linear.weight"])
    ir = _canonical_ir()

    monkeypatch.setattr(
        "mlxs.convert.planning._collect_runtime_tensor_schema",
        lambda _ir: [
            RuntimeTensorSchemaEntry(
                name="model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight",
                shape=(2, 2, 4),
            )
        ],
    )

    plan = build_conversion_plan(inspection, ir, options=ConversionOptions())

    assert len(plan.mappings) == 1
    mapping = plan.mappings[0]
    assert mapping.note == "switch_mlp_input_linear_split"
    assert mapping.transforms[0].kind.value == "slice"
    assert mapping.rule_id == "switch_mlp_input_linear_split"
    assert mapping.match_layer == "structural"
    assert mapping.adapter_name is None
