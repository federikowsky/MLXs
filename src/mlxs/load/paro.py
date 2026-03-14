"""PARO (paroquant) weight format loader — INT4 via paroquant library (FR1, §7.2).

Loads models from PARO checkpoints (e.g. z-lab/Qwen3.5-9B-PARO) using
paroquant's MLX backend. Adapts paroquant's (model, processor) to
ModelProtocol and TokenizerWrapper for use with generate/server.

Optional dependency: pip install "paroquant[mlx]" or mlxs[paro].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._errors import ModelLoadError
from mlxs.config.schema import ModelConfig
from mlxs.load.resolve import resolve_model_path
from mlxs.load.tokenizer import TokenizerWrapper
from mlxs.protocols.cache import CacheProtocol

logger = logging.getLogger(__name__)


def _detect_paro(model_path: Path) -> bool:
    """Return True if checkpoint is PARO (from config.json only, no paroquant import)."""
    from mlxs.load.weights import load_config

    try:
        config = load_config(model_path)
    except FileNotFoundError:
        return False
    return config.get("quantization_config", {}).get("quant_method") == "paroquant"


def load_paro_model_and_tokenizer(
    model_path: str | Path,
    model_config: ModelConfig,
    *,
    lazy: bool = False,
) -> tuple[nn.Module, TokenizerWrapper]:
    """Load a PARO-quantized model and tokenizer via paroquant's MLX backend.

    model_path can be a local directory or a Hugging Face model id; when not
    a directory, it is resolved via resolve_model_path using model_config
    (revision/token). Returns (model_adapter, tokenizer_wrapper) for use with
    generate/server. Raises ModelLoadError if paroquant is not installed or load fails.
    """
    try:
        from paroquant.inference.backends.mlx.load import load as paro_load
    except ImportError as e:
        raise ModelLoadError(
            "PARO weight format requires 'paroquant[mlx]'. "
            "Install with: pip install 'paroquant[mlx]' or pip install 'mlxs[paro]'."
        ) from e

    path = Path(model_path)
    if not path.is_dir():
        model_path = resolve_model_path(
            model_path,
            revision=model_config.model_hf_revision,
            token=model_config.model_hf_token,
        )
    path_str = str(model_path)

    try:
        model, processor, _ = paro_load(
            path_str,
            lazy=lazy,
            force_text=True,
        )
    except Exception as exc:
        raise ModelLoadError(f"PARO load failed for {model_path}: {exc}") from exc

    tokenizer = getattr(processor, "tokenizer", processor)
    adapter = _ParoModelAdapter(model)
    wrapper = TokenizerWrapper(tokenizer)
    logger.info("Loaded PARO model and tokenizer from %s", path_str)
    return adapter, wrapper


class _ParoModelAdapter(nn.Module):
    """Adapts paroquant (mlx_lm) model to ModelProtocol.

    Delegates __call__, make_cache, num_layers, vocab_size, parameters
    to the inner model so generate/batch see a single contract.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._inner = inner

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[CacheProtocol] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        # mlx_lm models build mask internally; do not pass mask (unsupported kwarg).
        return self._inner(input_ids, cache=cache)

    def make_cache(self) -> list[CacheProtocol]:
        return self._inner.make_cache()

    @property
    def num_layers(self) -> int:
        return self._inner.num_layers

    @property
    def vocab_size(self) -> int:
        return self._inner.vocab_size

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Delegate to inner model; identity if inner has no sanitize (§7.3)."""
        if hasattr(self._inner, "sanitize"):
            return self._inner.sanitize(weights)
        return weights

    def parameters(self) -> dict[str, Any]:
        """Delegate to inner model (nn.Module interface)."""
        return self._inner.parameters()
