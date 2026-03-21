from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlxs.convert.errors import ConversionVerificationError
from mlxs.convert.runtime_schema import runtime_schema_hash
from mlxs.convert.types import (
    ConversionOptions,
    ConversionPlan,
    ExecutionResult,
    MacroTemplate,
    RuntimeTensorSchemaEntry,
    SourceKind,
    TensorInfo,
    TensorTargetPlan,
    VerificationMode,
)
from mlxs.convert.verification import (
    _check_runtime_smoke_load,
    verify_conversion,
    verify_existing_output,
)


def _write_output_dir(
    output_dir: Path,
    *,
    config: dict[str, object] | None = None,
    tensors: dict[str, np.ndarray] | None = None,
    shard_name: str = "model.safetensors",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config or {"model_type": "qwen3"}))
    save_file(
        tensors
        or {"model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32)},
        str(output_dir / shard_name),
    )


def _inspection(output_dir: Path):
    return type(
        "Inspection",
        (),
        {
            "source_kind": SourceKind.LOCAL,
            "source_id": "fixture",
            "resolved_path": output_dir,
            "config_path": output_dir / "config.json",
            "config": {"model_type": "qwen3"},
            "weight_format": "safetensors",
            "sharded": False,
            "shard_files": ("model.safetensors",),
            "tensor_infos": (
                TensorInfo(
                    name="model.embed_tokens.weight",
                    shape=(2, 2),
                    dtype="F32",
                    file="model.safetensors",
                ),
            ),
            "tokenizer_artifacts": ("tokenizer.json",),
            "multimodal_artifacts": (),
            "custom_code_indicators": (),
            "warnings": (),
        },
    )()


def _plan(*, mode: VerificationMode) -> ConversionPlan:
    base_policy = (
        "schema",
        "weight_index",
        "config_invariants",
        "required_tensor_coverage",
        "shape",
        "schema_hash",
        "artifacts",
    )
    if mode == VerificationMode.REQUIRED:
        verification_policy = (*base_policy, "runtime_smoke")
    elif mode == VerificationMode.BASIC:
        verification_policy = base_policy
    else:
        verification_policy = ()

    target_schema = (
        RuntimeTensorSchemaEntry(
            name="model.embed_tokens.weight",
            shape=(2, 2),
            dtype="F32",
        ),
    )
    return ConversionPlan(
        macro_template=MacroTemplate.DECODER_DENSE,
        runtime_target_model_type="qwen3",
        runtime_model_mode="text",
        selected_profile="runtime_native",
        mappings=(
            TensorTargetPlan(
                target_name="model.embed_tokens.weight",
                source_names=("model.embed_tokens.weight",),
            ),
        ),
        skipped_source_tensors=(),
        target_schema=target_schema,
        required_target_names=("model.embed_tokens.weight",),
        selected_rules=("runtime_native",),
        verification_policy=verification_policy,
        normalized_config={"model_type": "qwen3"},
        target_schema_hash=runtime_schema_hash(target_schema),
    )


def _execution(output_dir: Path) -> ExecutionResult:
    return ExecutionResult(
        output_dir=output_dir,
        weight_files=("model.safetensors",),
        weight_index_file=None,
        written_tensor_names=("model.embed_tokens.weight",),
        copied_artifacts=("tokenizer.json",),
        skipped_source_tensors=(),
    )


def test_runtime_smoke_load_is_non_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_model(path: Path, *, lazy: bool = False, model_mode=None):
        calls.append({"path": path, "lazy": lazy, "model_mode": model_mode})
        return object()

    monkeypatch.setattr("mlxs.load.loader.load_model", fake_load_model)

    checks = []
    _check_runtime_smoke_load(tmp_path, "text", checks)

    assert calls
    assert calls[0]["lazy"] is False


def test_verify_existing_output_skip_returns_skipped(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_output_dir(output_dir)

    report = verify_existing_output(
        output_dir,
        options=ConversionOptions(verification_mode=VerificationMode.SKIP),
    )

    assert report.status.value == "skipped"


def test_verify_conversion_basic_does_not_call_runtime_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    _write_output_dir(output_dir)
    (output_dir / "tokenizer.json").write_text("{}")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("runtime smoke should not run in BASIC mode")

    monkeypatch.setattr("mlxs.load.loader.load_model", fail_if_called)

    report = verify_conversion(
        _inspection(output_dir),
        object(),  # unused by verify_conversion
        _plan(mode=VerificationMode.BASIC),
        _execution(output_dir),
        options=ConversionOptions(verification_mode=VerificationMode.BASIC),
    )

    assert report.status.value == "passed"


def test_verify_existing_output_uses_manifest_artifact_snapshot(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_output_dir(output_dir)

    manifest = {
        "runtime_model_mode": "text",
        "required_target_names": ["model.embed_tokens.weight"],
        "target_schema_hash": runtime_schema_hash(
            (
                RuntimeTensorSchemaEntry(
                    name="model.embed_tokens.weight",
                    shape=(2, 2),
                    dtype="F32",
                ),
            )
        ),
        "target_schema_snapshot": [
            {"name": "model.embed_tokens.weight", "shape": [2, 2], "dtype": "F32"}
        ],
        "normalized_config_snapshot": {"model_type": "qwen3"},
        "tokenizer_artifacts": ["tokenizer.json"],
        "multimodal_artifacts": [],
        "weight_files": ["model.safetensors"],
        "weight_index_file": None,
    }
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ConversionVerificationError, match="Verification failed"):
        verify_existing_output(
            output_dir,
            options=ConversionOptions(verification_mode=VerificationMode.BASIC),
        )


def test_verify_existing_output_accepts_adapter_mapping_provenance(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_output_dir(output_dir)
    (output_dir / "tokenizer.json").write_text("{}")

    snapshot = (
        RuntimeTensorSchemaEntry(name="model.embed_tokens.weight", shape=(2, 2), dtype="F32"),
    )
    manifest = {
        "runtime_model_mode": "text",
        "architecture_traits": {
            "modality": "text",
            "topology_kind": "decoder",
            "expert_layout": "dense",
            "sequence_family": "attention",
        },
        "required_target_names": [entry.name for entry in snapshot],
        "target_schema_hash": runtime_schema_hash(snapshot),
        "target_schema_snapshot": [
            {"name": entry.name, "shape": list(entry.shape), "dtype": entry.dtype}
            for entry in snapshot
        ],
        "normalized_config_snapshot": {"model_type": "qwen3"},
        "tokenizer_artifacts": ["tokenizer.json"],
        "multimodal_artifacts": [],
        "weight_files": ["model.safetensors"],
        "weight_index_file": None,
        "mapping_provenance": [
            {
                "target_name": "model.embed_tokens.weight",
                "source_names": ["language_model.model.embed_tokens.weight"],
                "rule_id": "qwen_family:language_model_prefix",
                "match_layer": "family_adapter",
                "adapter_name": "qwen_family",
                "required": True,
                "note": "pilot family alias",
                "transforms": [],
            }
        ],
    }
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest))

    report = verify_existing_output(
        output_dir,
        options=ConversionOptions(verification_mode=VerificationMode.BASIC),
    )

    assert report.status.value == "passed"


def test_verify_existing_output_accepts_qwen35_moe_mapping_provenance(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_output_dir(
        output_dir,
        config={"model_type": "qwen3_5_moe"},
        tensors={
            "language_model.model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32),
        },
    )
    (output_dir / "tokenizer.json").write_text("{}")

    snapshot = (
        RuntimeTensorSchemaEntry(
            name="language_model.model.embed_tokens.weight",
            shape=(2, 2),
            dtype="F32",
        ),
    )
    manifest = {
        "runtime_model_mode": "text",
        "architecture_traits": {
            "modality": "text",
            "topology_kind": "decoder",
            "expert_layout": "moe",
            "sequence_family": "ssm_hybrid",
        },
        "required_target_names": [entry.name for entry in snapshot],
        "target_schema_hash": runtime_schema_hash(snapshot),
        "target_schema_snapshot": [
            {"name": entry.name, "shape": list(entry.shape), "dtype": entry.dtype}
            for entry in snapshot
        ],
        "normalized_config_snapshot": {"model_type": "qwen3_5_moe"},
        "tokenizer_artifacts": ["tokenizer.json"],
        "multimodal_artifacts": [],
        "weight_files": ["model.safetensors"],
        "weight_index_file": None,
        "mapping_provenance": [
            {
                "target_name": "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                "source_names": [
                    "language_model.model.layers.0.mlp.experts.gate_up_proj.weight"
                ],
                "rule_id": "qwen35_moe_family:gate_up_split",
                "match_layer": "family_adapter",
                "adapter_name": "qwen35_moe_family",
                "required": True,
                "note": "qwen35_moe_gate_up_split",
                "transforms": [
                    {
                        "kind": "slice",
                        "axis": -2,
                        "slice_start": 0,
                        "slice_stop": None,
                    }
                ],
            }
        ],
    }
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest))

    report = verify_existing_output(
        output_dir,
        options=ConversionOptions(verification_mode=VerificationMode.BASIC),
    )

    assert report.status.value == "passed"


def test_verify_existing_output_requires_shard_index_from_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_output_dir(
        output_dir,
        tensors={"model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32)},
        shard_name="model-00001-of-00002.safetensors",
    )
    save_file(
        {"model.norm.weight": np.ones((2,), dtype=np.float32)},
        str(output_dir / "model-00002-of-00002.safetensors"),
    )

    snapshot = (
        RuntimeTensorSchemaEntry(name="model.embed_tokens.weight", shape=(2, 2), dtype="F32"),
        RuntimeTensorSchemaEntry(name="model.norm.weight", shape=(2,), dtype="F32"),
    )
    manifest = {
        "runtime_model_mode": "text",
        "required_target_names": [entry.name for entry in snapshot],
        "target_schema_hash": runtime_schema_hash(snapshot),
        "target_schema_snapshot": [
            {"name": entry.name, "shape": list(entry.shape), "dtype": entry.dtype}
            for entry in snapshot
        ],
        "normalized_config_snapshot": {"model_type": "qwen3"},
        "tokenizer_artifacts": [],
        "multimodal_artifacts": [],
        "weight_files": [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        "weight_index_file": "model.safetensors.index.json",
    }
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ConversionVerificationError, match="Verification failed"):
        verify_existing_output(
            output_dir,
            options=ConversionOptions(verification_mode=VerificationMode.BASIC),
        )
