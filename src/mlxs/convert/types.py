from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SourceKind(str, Enum):
    LOCAL = "local"
    HF_REPO = "hf_repo"


class MacroTemplate(str, Enum):
    DECODER_DENSE = "decoder_dense"
    DECODER_MOE = "decoder_moe"
    ENCODER_DECODER = "encoder_decoder"
    MULTIMODAL_DECODER = "multimodal_decoder"
    MULTIMODAL_ENCODER_DECODER = "multimodal_encoder_decoder"
    SSM_HYBRID = "ssm_hybrid"


class Modality(str, Enum):
    TEXT = "text"
    MULTIMODAL = "multimodal"


class DensityKind(str, Enum):
    DENSE = "dense"
    MOE = "moe"
    HYBRID = "hybrid"


class ConversionPhase(str, Enum):
    INSPECTION = "inspection"
    NORMALIZATION = "normalization"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"


class VerificationMode(str, Enum):
    REQUIRED = "required"
    BASIC = "basic"
    SKIP = "skip"


class ModelAssistanceMode(str, Enum):
    DISABLED = "disabled"
    OPTIONAL = "optional"


class TensorTransformKind(str, Enum):
    CAST = "cast"
    CONCAT = "concat"
    MOVE_AXIS = "move_axis"
    RESHAPE = "reshape"
    SLICE = "slice"
    STACK = "stack"
    TRANSPOSE = "transpose"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ConversionOptions:
    revision: str | None = None
    token: str | None = None
    dtype_policy: str = "preserve"
    target_format_version: str = "mlxs-converter-v1"
    quantization_mode: str | None = None
    max_shard_bytes: int | None = None
    verification_mode: VerificationMode = VerificationMode.REQUIRED
    model_assistance: ModelAssistanceMode = ModelAssistanceMode.DISABLED
    fail_on_custom_code: bool = True
    copy_tokenizer_artifacts: bool = True


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    file: str


@dataclass(frozen=True, slots=True)
class InspectionReport:
    source_kind: SourceKind
    source_id: str
    resolved_path: Path
    config_path: Path
    config: dict[str, Any]
    weight_format: str
    sharded: bool
    shard_files: tuple[str, ...]
    tensor_infos: tuple[TensorInfo, ...]
    tokenizer_artifacts: tuple[str, ...]
    multimodal_artifacts: tuple[str, ...]
    custom_code_indicators: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IRSource:
    kind: SourceKind
    source_id: str
    resolved_path: Path
    weight_format: str
    sharded: bool
    shard_list: tuple[str, ...]
    source_config_artifacts_present: tuple[str, ...]
    tokenizer_artifacts_present: tuple[str, ...]
    multimodal_artifacts_present: tuple[str, ...]
    custom_code_indicators: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IRIdentity:
    macro_template: MacroTemplate
    modality: Modality
    architecture_label: str
    variant_label: str | None
    runtime_target_model_type: str
    supported_by_runtime: bool


@dataclass(frozen=True, slots=True)
class IRTopology:
    backbone_type: str
    decoder_only: bool
    encoder_decoder: bool
    multimodal: bool
    ssm_hybrid: bool
    density: DensityKind
    attention_presence: bool
    attention_variant: str | None
    qkv_layout: str | None
    mlp_variant: str | None
    norm_type: str | None
    rope_variant: str | None
    embedding_tying: bool | None
    projector_presence: bool
    tower_presence: bool
    recurrent_traits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IRConfig:
    values: dict[str, Any]
    raw_config: dict[str, Any]
    required_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IRTensorLayout:
    selected_source_tensor_profile: str
    canonical_internal_tensor_profile: str
    tensor_groups_present: tuple[str, ...]
    optional_tensor_groups_present: tuple[str, ...]
    missing_required_groups: tuple[str, ...]
    fused_split_markers: tuple[str, ...]
    naming_aliases_discovered: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IRTokenizer:
    tokenizer_type: str
    artifact_paths: tuple[str, ...]
    special_tokens: dict[str, Any]
    vocab_metadata: dict[str, Any]
    chat_template_present: bool


@dataclass(frozen=True, slots=True)
class IRConversion:
    dtype_policy: str
    target_format_version: str
    quantization_mode: str | None
    shard_output_policy: str
    cast_rules: tuple[str, ...]
    save_options: dict[str, Any]
    canonical_output_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IREvidence:
    matched_config_keys: tuple[str, ...]
    matched_tensor_patterns: tuple[str, ...]
    matched_shape_traits: tuple[str, ...]
    matched_tokenizer_traits: tuple[str, ...]
    selected_rules: tuple[str, ...]
    model_assisted_normalization_usage: str | None


@dataclass(frozen=True, slots=True)
class IRAmbiguities:
    ambiguity_flags: tuple[str, ...]
    selected_resolution_path: str
    confidence_score: float | None
    fallback_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalIR:
    source: IRSource
    identity: IRIdentity
    topology: IRTopology
    config: IRConfig
    tensor_layout: IRTensorLayout
    tokenizer: IRTokenizer
    conversion: IRConversion
    evidence: IREvidence
    ambiguities: IRAmbiguities


@dataclass(frozen=True, slots=True)
class RuntimeTensorSchemaEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str | None = None


@dataclass(frozen=True, slots=True)
class TensorTransform:
    kind: TensorTransformKind
    axis: int | None = None
    permutation: tuple[int, ...] | None = None
    source_axis: int | None = None
    target_axis: int | None = None
    shape: tuple[int, ...] | None = None
    slice_start: int | None = None
    slice_stop: int | None = None
    dtype: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class TensorTargetPlan:
    target_name: str
    source_names: tuple[str, ...]
    transforms: tuple[TensorTransform, ...] = ()
    required: bool = True
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    macro_template: MacroTemplate
    runtime_target_model_type: str
    runtime_model_mode: str
    selected_profile: str
    mappings: tuple[TensorTargetPlan, ...]
    skipped_source_tensors: tuple[str, ...]
    target_schema: tuple[RuntimeTensorSchemaEntry, ...]
    required_target_names: tuple[str, ...]
    selected_rules: tuple[str, ...]
    verification_policy: tuple[str, ...]
    normalized_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    output_dir: Path
    weight_files: tuple[str, ...]
    weight_index_file: str | None
    written_tensor_names: tuple[str, ...]
    copied_artifacts: tuple[str, ...]
    skipped_source_tensors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    name: str
    status: VerificationStatus
    detail: str


@dataclass(frozen=True, slots=True)
class VerificationReport:
    status: VerificationStatus
    checks: tuple[VerificationCheck, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConversionManifest:
    source_identifier: str
    macro_template: str | None
    architecture_label: str | None
    runtime_target_model_type: str | None
    structural_parameters: dict[str, Any]
    model_assisted_normalization_used: bool
    tensor_profile: str | None
    required_tensor_count: int
    mapped_tensor_count: int
    skipped_tensor_count: int
    missing_tensor_count: int
    transforms: tuple[str, ...]
    output_format_version: str | None
    quantization_mode: str | None
    verification_status: str
    verification_checks: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    failure_phase: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ConversionResult:
    inspection: InspectionReport
    canonical_ir: CanonicalIR
    plan: ConversionPlan
    execution: ExecutionResult
    verification: VerificationReport
    manifest_path: Path


@dataclass(slots=True)
class PlanningContext:
    source_tensors: dict[str, TensorInfo]
    target_schema: dict[str, RuntimeTensorSchemaEntry]
    used_source_names: set[str] = field(default_factory=set)
