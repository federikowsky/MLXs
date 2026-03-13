"""Dependency wiring — composition root for the server (§9).

This is the ONLY place where concrete implementations are wired
together. All other modules depend only on protocols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.config.schema import AppConfig
from mlxs.generate import generate
from mlxs.generate.compile import warmup
from mlxs.load import load_model_and_tokenizer
from mlxs.load.tokenizer import TokenizerWrapper
from mlxs.observability.logger import setup_logging
from mlxs.observability.metrics import create_metrics
from mlxs.prompt_cache import PromptCache

logger = logging.getLogger(__name__)


@dataclass
class Dependencies:
    """Container for all wired dependencies."""

    config: AppConfig
    model: nn.Module
    tokenizer: TokenizerWrapper
    prompt_cache: PromptCache
    metrics: Any
    generate_fn: Any  # The generate function


def create_dependencies(config: AppConfig) -> Dependencies:
    """Wire all dependencies from config. Called once at startup.

    This is the composition root (§9). Concrete implementations are
    resolved here and passed to the server.
    """
    # Setup observability
    setup_logging(
        level=config.observability.log_level,
        json_format=False,
    )
    metrics = create_metrics(config.observability)

    # Set wired limit
    if config.memory.wired_limit is not None:
        mx.set_wired_limit(config.memory.wired_limit)
    else:
        try:
            info = mx.device_info()
            recommended = info.get("max_recommended_working_set_size")
            if recommended:
                mx.set_wired_limit(recommended)
                logger.info("Set wired limit to %d bytes (device recommendation)", recommended)
        except Exception:
            pass

    # Load model and tokenizer (format-aware: safetensors, PARO, etc.)
    logger.info("Loading model from %s", config.model.model_path)
    model, tokenizer = load_model_and_tokenizer(
        config.model.model_path,
        config.model,
    )

    # Warmup if configured
    if config.generate.warmup_after_load:
        warmup(model, model.make_cache, vocab_size=model.vocab_size)

    # Create prompt cache
    prompt_cache = PromptCache(config.prompt_cache)

    return Dependencies(
        config=config,
        model=model,
        tokenizer=tokenizer,
        prompt_cache=prompt_cache,
        metrics=metrics,
        generate_fn=generate,
    )
