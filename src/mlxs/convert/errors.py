from __future__ import annotations

from typing import Any

from mlxs.convert.types import ConversionPhase


class ConverterError(Exception):
    code = "converter_error"

    def __init__(
        self,
        message: str,
        *,
        phase: ConversionPhase,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.phase = phase
        self.details = details or {}
        super().__init__(message)


class UnsupportedSourceFormatError(ConverterError):
    code = "unsupported_source_format"


class UnsupportedTopologyError(ConverterError):
    code = "unsupported_topology"


class UnsupportedAmbiguityError(ConverterError):
    code = "unsupported_ambiguity"


class MissingRequiredConfigError(ConverterError):
    code = "missing_required_config"


class MissingRequiredTensorError(ConverterError):
    code = "missing_required_tensors"


class InvalidTransformRequestError(ConverterError):
    code = "invalid_transform_request"


class IncompatibleTensorShapeError(ConverterError):
    code = "incompatible_tensor_shape"


class SerializationFailureError(ConverterError):
    code = "serialization_failure"


class ConversionVerificationError(ConverterError):
    code = "verification_failure"


class UnsupportedRuntimeTargetError(ConverterError):
    code = "unsupported_runtime_target"
