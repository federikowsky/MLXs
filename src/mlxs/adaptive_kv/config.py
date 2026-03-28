"""Adaptive KV TurboQuant-first configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class AdaptiveKVConfig(BaseModel):
    """Config surface for the TurboQuant-first Adaptive KV branch."""

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
        description="Decode steps between control-path profile updates.",
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
        description="Most recent blocks protected from ordinary degradation/eviction.",
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
    t_tq_safe_restore: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for TQ_AGGR -> TQ_SAFE restoration.",
    )
    t_tq_safe_degrade: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Threshold for TQ_SAFE -> TQ_AGGR degradation.",
    )
    t_evict_candidate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Candidate threshold for TQ_AGGR -> EVICTED.",
    )
    min_dwell_safe: int = Field(
        default=1,
        ge=0,
        description="Minimum control windows to remain TQ_SAFE before degradation.",
    )
    min_dwell_aggr: int = Field(
        default=1,
        ge=0,
        description="Minimum control windows to remain TQ_AGGR before restore/eviction.",
    )
    restore_cooldown: int = Field(
        default=1,
        ge=0,
        description="Control windows to wait after a TQ_SAFE -> TQ_AGGR degradation.",
    )
    degrade_cooldown: int = Field(
        default=1,
        ge=0,
        description="Control windows to wait after a TQ_AGGR -> TQ_SAFE restore.",
    )
    tq_safe_bits: int = Field(
        default=8,
        ge=1,
        le=8,
        description="TurboQuant bitwidth for the TQ_SAFE resident profile.",
    )
    tq_aggr_bits: int = Field(
        default=4,
        ge=1,
        le=8,
        description="TurboQuant bitwidth for the TQ_AGGR resident profile.",
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
        if self.t_tq_safe_restore <= self.t_tq_safe_degrade:
            raise ValueError(
                "t_tq_safe_restore must be > t_tq_safe_degrade for profile hysteresis"
            )
        if self.tq_safe_bits < self.tq_aggr_bits:
            raise ValueError("tq_safe_bits must be >= tq_aggr_bits")
        return self
