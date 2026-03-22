"""Adaptive score update logic."""

from __future__ import annotations

from dataclasses import replace

from mlxs.adaptive_kv.block_types import BlockRecord, ScoreComponents
from mlxs.adaptive_kv.config import AdaptiveKVConfig


class AdaptiveScoreEngine:
    """Updates block scores from usage and structural priors."""

    def __init__(self, config: AdaptiveKVConfig) -> None:
        self._config = config

    def update(self, block: BlockRecord, usage: float, *, step: int) -> BlockRecord:
        smoothed_usage = (
            self._config.usage_alpha * usage
            + (1.0 - self._config.usage_alpha) * block.score.last_usage
        )
        hotness = self._ema(block.score.hotness, smoothed_usage, self._config.rho_hot)
        persistence = self._ema(
            block.score.persistence,
            smoothed_usage,
            self._config.rho_persist,
        )
        age_windows = 0 if smoothed_usage > 0 else block.age_windows + 1
        age_penalty = min(1.0, age_windows * self._config.age_lambda)
        raw_score = (
            self._config.w_hot * hotness
            + self._config.w_persist * persistence
            + self._config.w_struct * block.structural_prior
            - self._config.w_age * age_penalty
        )
        score = ScoreComponents(
            hotness=self._clamp(hotness),
            persistence=self._clamp(persistence),
            structural_prior=self._clamp(block.structural_prior),
            age_penalty=self._clamp(age_penalty),
            composite=self._clamp(raw_score),
            last_usage=self._clamp(smoothed_usage),
        )
        return replace(
            block,
            age_windows=age_windows,
            windows_in_tier=block.windows_in_tier + 1,
            last_access_step=step if smoothed_usage > 0 else block.last_access_step,
            score=score,
        )

    def _ema(self, current: float, value: float, rho: float) -> float:
        return (1.0 - rho) * current + rho * value

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

