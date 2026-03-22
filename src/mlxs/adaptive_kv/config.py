"""Adaptive KV configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class AdaptiveKVConfig(BaseModel):
    """Config surface for the adaptive KV subsystem."""

    model_config = {"frozen": True, "extra": "forbid"}

    enabled: bool = Field(default=False, description="Enable adaptive KV mode.")
    block_size_tokens: int = Field(
        default=64,
        ge=1,
        description="Logical token span per adaptive block.",
    )
    update_window_steps: int = Field(
        default=16,
        ge=1,
        description="Decode steps between control-path policy updates.",
    )
    soft_budget_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Soft resident-memory budget for adaptive KV.",
    )
    hard_budget_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Hard resident-memory budget for adaptive KV.",
    )
    recent_tail_protect_blocks: int = Field(
        default=2,
        ge=0,
        description="Most recent blocks protected from ordinary eviction.",
    )
    usage_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Smoothing factor for windowed usage updates.",
    )
    rho_hot: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="EMA weight for hotness updates.",
    )
    rho_persist: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="EMA weight for persistence updates.",
    )
    age_lambda: float = Field(
        default=0.1,
        ge=0.0,
        description="Per-window age penalty increment.",
    )
    w_hot: float = Field(default=0.4, ge=0.0, le=1.0, description="Hotness score weight.")
    w_persist: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Persistence score weight.",
    )
    w_struct: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Structural-prior score weight.",
    )
    w_age: float = Field(default=0.1, ge=0.0, le=1.0, description="Age-penalty score weight.")
    t_full_promote: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Promotion threshold for COMPRESSED -> FULL.",
    )
    t_full_demote: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Demotion threshold for FULL -> COMPRESSED.",
    )
    t_evict_candidate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Candidate threshold for COMPRESSED -> EVICTED.",
    )
    min_dwell_full: int = Field(
        default=1,
        ge=0,
        description="Minimum control windows to remain FULL before demotion.",
    )
    min_dwell_compressed: int = Field(
        default=1,
        ge=0,
        description="Minimum control windows to remain COMPRESSED before promotion/eviction.",
    )
    promote_cooldown: int = Field(
        default=1,
        ge=0,
        description="Control windows to wait after a promotion.",
    )
    demote_cooldown: int = Field(
        default=1,
        ge=0,
        description="Control windows to wait after a demotion.",
    )
    diagnostics_enabled: bool = Field(
        default=True,
        description="Enable debug snapshots and per-block diagnostics.",
    )
    emit_metrics: bool = Field(
        default=True,
        description="Emit adaptive metrics through the configured sink.",
    )

    @model_validator(mode="after")
    def _validate_budgets(self) -> AdaptiveKVConfig:
        if (
            self.soft_budget_bytes is not None
            and self.hard_budget_bytes is not None
            and self.hard_budget_bytes < self.soft_budget_bytes
        ):
            raise ValueError("hard_budget_bytes must be >= soft_budget_bytes")
        weight_sum = self.w_hot + self.w_persist + self.w_struct + self.w_age
        if weight_sum <= 0:
            raise ValueError("Adaptive KV score weights must sum to a positive value")
        if self.t_full_promote <= self.t_full_demote:
            raise ValueError("t_full_promote must be > t_full_demote for hysteresis")
        return self
