from __future__ import annotations

import mlx.core as mx

from mlxs.adaptive_kv.usage import AdaptiveUsageCollector


def test_snapshot_and_reset_normalizes_deferred_device_batches() -> None:
    collector = AdaptiveUsageCollector()
    collector.record_batch((1, 2), mx.array([2.0, 1.0], dtype=mx.float32))
    collector.record_batch((1,), mx.array([1.0], dtype=mx.float32))

    usage = collector.snapshot_and_reset()

    assert usage == {1: 1.0, 2: 1.0 / 3.0}


def test_snapshot_and_reset_merges_host_and_device_usage() -> None:
    collector = AdaptiveUsageCollector()
    collector.record(1, 2.0)
    collector.record_batch((1, 2), mx.array([1.0, 2.0], dtype=mx.float32))

    usage = collector.snapshot_and_reset()

    assert usage == {1: 1.0, 2: 2.0 / 3.0}


def test_snapshot_and_reset_clears_pending_batches() -> None:
    collector = AdaptiveUsageCollector()
    collector.record_batch((1,), mx.array([1.0], dtype=mx.float32))

    assert collector.snapshot_and_reset() == {1: 1.0}
    assert collector.snapshot_and_reset() == {}
