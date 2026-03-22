from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from mlxs.convert.execution import (
    _apply_transform,
    _execute_weight_write,
    _load_source_weights,
    _materialize_target_tensors,
    _prepare_execution_dependencies,
)
from mlxs.convert.types import (
    ConversionPlan,
    InspectionReport,
    MacroTemplate,
    MappingDependencyGroup,
    OutputPackGroup,
    RuntimeTensorSchemaEntry,
    SourceKind,
    TensorInfo,
    TensorTargetPlan,
    TensorTransform,
    TensorTransformKind,
)


def _inspection(tmp_path: Path) -> InspectionReport:
    return InspectionReport(
        source_kind=SourceKind.LOCAL,
        source_id="fixture",
        resolved_path=tmp_path,
        config_path=tmp_path / "config.json",
        config={"model_type": "qwen3"},
        weight_format="safetensors",
        sharded=True,
        shard_files=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
        tensor_infos=(
            TensorInfo(
                name="model.embed_tokens.weight",
                shape=(2, 2),
                dtype="F32",
                file="model-00001-of-00002.safetensors",
            ),
            TensorInfo(
                name="model.layers.0.self_attn.q_proj.weight",
                shape=(2, 2),
                dtype="F32",
                file="model-00001-of-00002.safetensors",
            ),
            TensorInfo(
                name="model.layers.0.self_attn.k_proj.weight",
                shape=(2, 2),
                dtype="F32",
                file="model-00002-of-00002.safetensors",
            ),
            TensorInfo(
                name="unused.weight",
                shape=(2, 2),
                dtype="F32",
                file="model-00002-of-00002.safetensors",
            ),
        ),
        tokenizer_artifacts=("tokenizer.json",),
        multimodal_artifacts=(),
        custom_code_indicators=(),
    )


def _plan(mappings: tuple[TensorTargetPlan, ...]) -> ConversionPlan:
    return ConversionPlan(
        macro_template=MacroTemplate.DECODER_DENSE,
        runtime_target_model_type="qwen3",
        runtime_model_mode="text",
        selected_profile="runtime_native",
        mappings=mappings,
        skipped_source_tensors=("unused.weight",),
        target_schema=tuple(
            RuntimeTensorSchemaEntry(name=mapping.target_name, shape=(2, 2))
            for mapping in mappings
        ),
        required_target_names=tuple(mapping.target_name for mapping in mappings),
        selected_rules=("runtime_native",),
        verification_policy=("shape",),
        normalized_config={"model_type": "qwen3"},
    )


def test_apply_transform_stack() -> None:
    values = [np.ones((2, 2)), np.zeros((2, 2))]
    transform = TensorTransform(kind=TensorTransformKind.STACK, axis=0)

    result = _apply_transform(values, transform)

    assert result.shape == (2, 2, 2)


def test_apply_transform_slice_second_half() -> None:
    value = np.arange(12).reshape(6, 2)
    transform = TensorTransform(
        kind=TensorTransformKind.SLICE,
        axis=0,
        note="Split shared MLP input_linear second half",
    )

    result = _apply_transform(value, transform)

    assert result.shape == (3, 2)
    assert result[0, 0] == 6


def test_materialize_target_tensors_with_move_axis() -> None:
    source = {"backbone.layers.0.mixer.conv1d.weight": np.ones((4, 3, 1))}
    plan = ConversionPlan(
        macro_template=MacroTemplate.SSM_HYBRID,
        runtime_target_model_type="mamba2",
        runtime_model_mode="text",
        selected_profile="conv_axis_sensitive",
        mappings=(
            TensorTargetPlan(
                target_name="backbone.layers.0.mixer.conv1d.weight",
                source_names=("backbone.layers.0.mixer.conv1d.weight",),
                transforms=(
                    TensorTransform(
                        kind=TensorTransformKind.MOVE_AXIS,
                        source_axis=2,
                        target_axis=1,
                    ),
                ),
            ),
        ),
        skipped_source_tensors=(),
        target_schema=(
            RuntimeTensorSchemaEntry(
                name="backbone.layers.0.mixer.conv1d.weight",
                shape=(4, 1, 3),
            ),
        ),
        required_target_names=("backbone.layers.0.mixer.conv1d.weight",),
        selected_rules=("conv_axis_sensitive",),
        verification_policy=("shape",),
        normalized_config={"model_type": "mamba2"},
    )

    materialized = _materialize_target_tensors(source, plan)

    assert materialized["backbone.layers.0.mixer.conv1d.weight"].shape == (4, 1, 3)


def test_prepare_execution_dependencies_groups_referenced_sources_by_shard(
    tmp_path: Path,
) -> None:
    inspection = _inspection(tmp_path)
    plan = _plan(
        (
            TensorTargetPlan(
                target_name="model.embed_tokens.weight",
                source_names=("model.embed_tokens.weight",),
            ),
            TensorTargetPlan(
                target_name="model.layers.0.self_attn.qkv.weight",
                source_names=(
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                ),
                transforms=(TensorTransform(kind=TensorTransformKind.STACK, axis=0),),
            ),
        )
    )

    dependencies = _prepare_execution_dependencies(inspection, plan)

    assert dependencies.referenced_source_tensors == (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    )
    assert dependencies.referenced_source_shards == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    assert dependencies.source_shard_dependencies[0].source_names == (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
    )
    assert dependencies.source_shard_dependencies[0].target_names == (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.qkv.weight",
    )
    assert dependencies.source_shard_dependencies[1].source_names == (
        "model.layers.0.self_attn.k_proj.weight",
    )
    assert dependencies.mapping_dependency_groups == (
        MappingDependencyGroup(
            shard_files=("model-00001-of-00002.safetensors",),
            target_names=("model.embed_tokens.weight",),
        ),
        MappingDependencyGroup(
            shard_files=(
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ),
            target_names=("model.layers.0.self_attn.qkv.weight",),
        ),
    )


def test_load_source_weights_reads_only_referenced_tensors(
    tmp_path: Path,
) -> None:
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    shard2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.full((2, 2), 2.0, dtype=np.float32),
        },
        str(shard1),
    )
    save_file(
        {
            "model.layers.0.self_attn.k_proj.weight": np.full((2, 2), 3.0, dtype=np.float32),
            "unused.weight": np.full((2, 2), 9.0, dtype=np.float32),
        },
        str(shard2),
    )
    inspection = _inspection(tmp_path)
    plan = _plan(
        (
            TensorTargetPlan(
                target_name="model.embed_tokens.weight",
                source_names=("model.embed_tokens.weight",),
            ),
            TensorTargetPlan(
                target_name="model.layers.0.self_attn.k_proj.weight",
                source_names=("model.layers.0.self_attn.k_proj.weight",),
            ),
        )
    )

    dependencies = _prepare_execution_dependencies(inspection, plan)
    loaded, strategy = _load_source_weights(inspection, dependencies)

    assert strategy == "selective_safetensors"
    assert tuple(sorted(loaded)) == (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.k_proj.weight",
    )
    assert "unused.weight" not in loaded
    assert np.allclose(loaded["model.embed_tokens.weight"], np.ones((2, 2), dtype=np.float32))
    assert np.allclose(
        loaded["model.layers.0.self_attn.k_proj.weight"],
        np.full((2, 2), 3.0, dtype=np.float32),
    )


def test_execute_weight_write_single_file_tracks_pack_group_and_preserves_tensors(
    tmp_path: Path,
) -> None:
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    shard2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.full((2, 2), 2.0, dtype=np.float32),
        },
        str(shard1),
    )
    save_file(
        {
            "model.layers.0.self_attn.k_proj.weight": np.full((2, 2), 3.0, dtype=np.float32),
        },
        str(shard2),
    )
    inspection = _inspection(tmp_path)
    plan = _plan(
        (
            TensorTargetPlan(
                target_name="model.embed_tokens.weight",
                source_names=("model.embed_tokens.weight",),
            ),
            TensorTargetPlan(
                target_name="model.layers.0.self_attn.k_proj.weight",
                source_names=("model.layers.0.self_attn.k_proj.weight",),
            ),
        )
    )

    dependencies = _prepare_execution_dependencies(inspection, plan)
    output_dir = tmp_path / "output-single"
    output_dir.mkdir()

    outcome = _execute_weight_write(
        inspection,
        plan,
        output_dir,
        dependencies=dependencies,
        max_shard_bytes=None,
    )

    assert outcome.load_strategy == "selective_safetensors"
    assert outcome.materialization_strategy == "single_file_buffered"
    assert outcome.weight_files == ("model.safetensors",)
    assert outcome.weight_index_file is None
    assert outcome.output_pack_groups == (
        OutputPackGroup(
            target_names=(
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ),
            total_bytes=32,
        ),
    )

    written = load_file(str(output_dir / "model.safetensors"))
    assert tuple(sorted(written)) == (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.k_proj.weight",
    )
    assert np.allclose(written["model.embed_tokens.weight"], np.ones((2, 2), dtype=np.float32))
    assert np.allclose(
        written["model.layers.0.self_attn.k_proj.weight"],
        np.full((2, 2), 3.0, dtype=np.float32),
    )


def test_execute_weight_write_sharded_flushes_by_pack_group_and_writes_index(
    tmp_path: Path,
) -> None:
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    shard2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.full((2, 2), 2.0, dtype=np.float32),
        },
        str(shard1),
    )
    save_file(
        {
            "model.layers.0.self_attn.k_proj.weight": np.full((2, 2), 3.0, dtype=np.float32),
        },
        str(shard2),
    )
    inspection = _inspection(tmp_path)
    plan = _plan(
        (
            TensorTargetPlan(
                target_name="model.embed_tokens.weight",
                source_names=("model.embed_tokens.weight",),
            ),
            TensorTargetPlan(
                target_name="model.layers.0.self_attn.q_proj.weight",
                source_names=("model.layers.0.self_attn.q_proj.weight",),
            ),
            TensorTargetPlan(
                target_name="model.layers.0.self_attn.k_proj.weight",
                source_names=("model.layers.0.self_attn.k_proj.weight",),
            ),
        )
    )

    dependencies = _prepare_execution_dependencies(inspection, plan)
    output_dir = tmp_path / "output-sharded"
    output_dir.mkdir()

    outcome = _execute_weight_write(
        inspection,
        plan,
        output_dir,
        dependencies=dependencies,
        max_shard_bytes=20,
    )

    assert outcome.load_strategy == "selective_safetensors"
    assert outcome.materialization_strategy == "incremental_shard_buffered"
    assert outcome.weight_files == (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    assert outcome.weight_index_file == "model.safetensors.index.json"
    assert outcome.output_pack_groups == (
        OutputPackGroup(target_names=("model.embed_tokens.weight",), total_bytes=16),
        OutputPackGroup(target_names=("model.layers.0.self_attn.k_proj.weight",), total_bytes=16),
        OutputPackGroup(target_names=("model.layers.0.self_attn.q_proj.weight",), total_bytes=16),
    )

    merged: dict[str, np.ndarray] = {}
    for file_name in outcome.weight_files:
        merged.update(load_file(str(output_dir / file_name)))
    assert tuple(sorted(merged)) == (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    )
    assert np.allclose(merged["model.embed_tokens.weight"], np.ones((2, 2), dtype=np.float32))
    assert np.allclose(
        merged["model.layers.0.self_attn.q_proj.weight"],
        np.full((2, 2), 2.0, dtype=np.float32),
    )
    assert np.allclose(
        merged["model.layers.0.self_attn.k_proj.weight"],
        np.full((2, 2), 3.0, dtype=np.float32),
    )

    index_payload = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index_payload["metadata"]["total_size"] == 48
    assert index_payload["weight_map"] == {
        "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
        "model.layers.0.self_attn.k_proj.weight": "model-00002-of-00003.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00003-of-00003.safetensors",
    }
