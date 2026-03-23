"""Adaptive KV metric helpers."""

from __future__ import annotations

from typing import Any

from mlxs.adaptive_kv.block_types import BlockRecord, BlockTier

BLOCKS_TOTAL = "adaptive_kv_blocks_total"
BLOCKS_FULL = "adaptive_kv_blocks_full"
BLOCKS_COMPRESSED = "adaptive_kv_blocks_compressed"
BLOCKS_EVICTED = "adaptive_kv_blocks_evicted"
PROMOTIONS_TOTAL = "adaptive_kv_promotions_total"
DEMOTIONS_TOTAL = "adaptive_kv_demotions_total"
EVICTIONS_TOTAL = "adaptive_kv_evictions_total"
RECOMPUTATIONS_TOTAL = "adaptive_kv_recomputations_total"
SCORE_UPDATES_TOTAL = "adaptive_kv_score_updates_total"
PRESSURE_SOFT_COUNT = "adaptive_kv_pressure_soft_count"
PRESSURE_HARD_COUNT = "adaptive_kv_pressure_hard_count"
POLICY_TIME_SECONDS = "adaptive_kv_policy_time_seconds"
COMPATIBILITY_FALLBACKS_TOTAL = "adaptive_kv_compatibility_fallbacks_total"
REPLAY_FORWARD_TIME_SECONDS_TOTAL = "adaptive_kv_replay_forward_seconds_total"
REPLAY_FORWARD_EVENTS_TOTAL = "adaptive_kv_replay_forward_events_total"
RECOVERY_MATERIALIZATION_TIME_SECONDS_TOTAL = (
    "adaptive_kv_recovery_materialization_seconds_total"
)
RECOVERY_MATERIALIZATION_EVENTS_TOTAL = "adaptive_kv_recovery_materialization_events_total"
POST_RECOVERY_DECODE_TIME_SECONDS_TOTAL = "adaptive_kv_post_recovery_decode_seconds_total"
POST_RECOVERY_DECODE_FORWARDS_TOTAL = "adaptive_kv_post_recovery_decode_forwards_total"


def emit_population(metrics: Any, blocks: list[BlockRecord]) -> None:
    metrics.gauge(BLOCKS_TOTAL, float(len(blocks)))
    metrics.gauge(BLOCKS_FULL, float(sum(block.tier is BlockTier.FULL for block in blocks)))
    metrics.gauge(
        BLOCKS_COMPRESSED,
        float(sum(block.tier is BlockTier.COMPRESSED for block in blocks)),
    )
    metrics.gauge(BLOCKS_EVICTED, float(sum(block.tier is BlockTier.EVICTED for block in blocks)))


def emit_compatibility_fallback(metrics: Any, *, reason: str) -> None:
    metrics.counter(COMPATIBILITY_FALLBACKS_TOTAL, reason=reason)


def block_debug_view(block: BlockRecord, *, ghost_present: bool) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "token_span": (block.start_token, block.end_token),
        "source_span": (block.source_start, block.source_end),
        "segment_id": block.segment_id,
        "tier": block.tier.value,
        "pin_state": block.pin_state.value,
        "score_components": {
            "hotness": block.score.hotness,
            "persistence": block.score.persistence,
            "structural_prior": block.score.structural_prior,
            "age_penalty": block.score.age_penalty,
            "composite": block.score.composite,
            "last_usage": block.score.last_usage,
        },
        "age_windows": block.age_windows,
        "last_transition": (
            {
                "step": block.last_transition.step,
                "from_tier": block.last_transition.from_tier.value,
                "to_tier": block.last_transition.to_tier.value,
                "reason": block.last_transition.reason,
            }
            if block.last_transition is not None
            else None
        ),
        "ghost_present": ghost_present,
    }
