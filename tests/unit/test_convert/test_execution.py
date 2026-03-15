from __future__ import annotations

from pathlib import Path

import numpy as np

from mlxs.convert.execution import _apply_transform, _materialize_target_tensors
from mlxs.convert.types import (
    ConversionPlan,
    MacroTemplate,
    RuntimeTensorSchemaEntry,
    TensorTargetPlan,
    TensorTransform,
    TensorTransformKind,
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
