"""Baseline labels and AdaptiveKVConfig presets for benchmarking."""

from __future__ import annotations

from typing import Literal

from mlxs.adaptive_kv.config import AdaptiveKVConfig

BaselineKind = Literal["non_adaptive", "adaptive_full", "adaptive_soft", "adaptive_hard"]

# Effectively unlimited for macOS-scale resident KV (plan: isolate control-path overhead).
_HUGE_BUDGET = 2**50


def adaptive_config_for_baseline(
    kind: BaselineKind,
    *,
    block_size_tokens: int = 64,
    update_window_steps: int = 16,
    soft_budget_bytes: int | None = None,
    hard_budget_bytes: int | None = None,
    budget_profile: str | None = None,
) -> AdaptiveKVConfig | None:
    """Return AdaptiveKVConfig for adaptive baselines, or None for non-adaptive."""
    if kind == "non_adaptive":
        return None

    base_kw: dict = {
        "enabled": True,
        "emit_metrics": True,
        "block_size_tokens": block_size_tokens,
        "update_window_steps": update_window_steps,
    }

    if kind == "adaptive_full":
        return AdaptiveKVConfig(
            **base_kw,
            soft_budget_bytes=_HUGE_BUDGET,
            hard_budget_bytes=_HUGE_BUDGET,
            t_tq_safe_degrade=0.0,
            t_evict_candidate=0.0,
        )

    soft, hard = _resolve_budget_pair(
        kind,
        soft_budget_bytes=soft_budget_bytes,
        hard_budget_bytes=hard_budget_bytes,
        budget_profile=budget_profile,
    )

    if kind == "adaptive_soft":
        return AdaptiveKVConfig(
            **base_kw,
            soft_budget_bytes=soft,
            hard_budget_bytes=hard,
        )

    if kind == "adaptive_hard":
        return AdaptiveKVConfig(
            **base_kw,
            soft_budget_bytes=soft,
            hard_budget_bytes=hard,
            recent_tail_protect_blocks=0,
            t_tq_safe_restore=0.95,
            t_tq_safe_degrade=0.9,
            t_evict_candidate=0.99,
        )

    raise ValueError(kind)


def _resolve_budget_pair(
    kind: BaselineKind,
    *,
    soft_budget_bytes: int | None,
    hard_budget_bytes: int | None,
    budget_profile: str | None,
) -> tuple[int, int]:
    if (soft_budget_bytes is None) ^ (hard_budget_bytes is None):
        raise ValueError("set both soft_budget_bytes and hard_budget_bytes or neither")
    if soft_budget_bytes is not None and hard_budget_bytes is not None:
        return soft_budget_bytes, hard_budget_bytes

    profile = budget_profile or "default"

    if profile == "tiny_stress":
        # Matches spirit of tests/unit/test_generate/test_adaptive_kv.py small Llama.
        return 500, 900

    if profile == "default":
        if kind == "adaptive_soft":
            # Compression-friendly soft cap; override bytes for your model.
            return 80_000_000, 500_000_000
        if kind == "adaptive_hard":
            return 50_000_000, 120_000_000

    raise ValueError(
        f"budget_profile={profile!r} missing or incomplete; "
        "pass --soft-budget-bytes and --hard-budget-bytes for your model"
    )
