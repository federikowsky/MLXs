from __future__ import annotations

from pathlib import Path

import pytest

from mlxs.convert.types import (
    CanonicalIR,
    ConversionOptions,
    DensityKind,
    IRAmbiguities,
    IRConfig,
    IRConversion,
    IREvidence,
    IRIdentity,
    IRSource,
    IRTensorLayout,
    IRTokenizer,
    IRTopology,
    Modality,
    SourceKind,
)
from mlxs.convert.verification import _check_runtime_smoke_load


def _canonical_ir(multimodal: bool = False) -> CanonicalIR:
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
            macro_template="multimodal_decoder" if multimodal else "decoder_dense",  # type: ignore[arg-type]
            modality=Modality.MULTIMODAL if multimodal else Modality.TEXT,
            architecture_label="qwen3",
            variant_label=None,
            runtime_target_model_type="qwen3",
            supported_by_runtime=True,
        ),
        topology=IRTopology(
            backbone_type="multimodal_decoder" if multimodal else "decoder",
            decoder_only=True,
            encoder_decoder=False,
            multimodal=multimodal,
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
            tower_presence=False,
            recurrent_traits=(),
        ),
        config=IRConfig(values={}, raw_config={"model_type": "qwen3"}, required_fields=("hidden_size",)),
        tensor_layout=IRTensorLayout(
            selected_source_tensor_profile="runtime_native",
            canonical_internal_tensor_profile="runtime_post_sanitize",
            tensor_groups_present=("core",),
            optional_tensor_groups_present=(),
            missing_required_groups=(),
            fused_split_markers=("runtime_native",),
            naming_aliases_discovered=(),
        ),
        tokenizer=IRTokenizer(
            tokenizer_type="tokenizer_json",
            artifact_paths=("tokenizer.json",),
            special_tokens={},
            vocab_metadata={"vocab_size": 32},
            chat_template_present=False,
        ),
        conversion=IRConversion(
            dtype_policy="preserve",
            target_format_version="mlxs-converter-v1",
            quantization_mode=None,
            shard_output_policy="single",
            cast_rules=("preserve",),
            save_options={},
            canonical_output_config={"model_type": "qwen3"},
        ),
        evidence=IREvidence(
            matched_config_keys=("model_type",),
            matched_tensor_patterns=("runtime_native",),
            matched_shape_traits=(),
            matched_tokenizer_traits=("tokenizer.json",),
            selected_rules=("runtime_native",),
            model_assisted_normalization_usage=None,
        ),
        ambiguities=IRAmbiguities(
            ambiguity_flags=(),
            selected_resolution_path="deterministic_rules",
            confidence_score=None,
            fallback_markers=(),
        ),
    )


def test_runtime_smoke_load_is_non_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_model(path: Path, *, lazy: bool = False, model_mode=None):
        calls.append({"path": path, "lazy": lazy, "model_mode": model_mode})
        return object()

    monkeypatch.setattr("mlxs.load.loader.load_model", fake_load_model)

    checks = []
    _check_runtime_smoke_load(_canonical_ir(), tmp_path, checks)

    assert calls
    assert calls[0]["lazy"] is False

