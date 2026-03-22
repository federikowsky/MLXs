from __future__ import annotations

import mlx.core as mx

from mlxs.adaptive_kv.storage import AdaptiveCompressedRunStore


def _full_state(length: int) -> tuple[mx.array, mx.array]:
    base = mx.arange(length * 32, dtype=mx.float32).reshape(1, 1, length, 32)
    return base, base + 100.0


def test_adjacent_compressed_runs_merge_with_block_local_slices() -> None:
    keys_a, values_a = _full_state(2)
    keys_b, values_b = _full_state(2)
    run_a = AdaptiveCompressedRunStore.from_full_block(
        0,
        block_id=10,
        keys=keys_a,
        values=values_a,
        group_size=32,
        bits=8,
    )
    run_b = AdaptiveCompressedRunStore.from_full_block(
        1,
        block_id=11,
        keys=keys_b,
        values=values_b,
        group_size=32,
        bits=8,
    )

    merged = run_a.merge_with(run_b, run_id=0)

    assert merged.token_count == 4
    assert merged.block_slices == ((10, 0, 2), (11, 2, 4))
    assert merged.block_live_bytes(10) > 0
    assert merged.block_live_bytes(11) > 0


def test_split_without_block_preserves_left_and_right_local_offsets() -> None:
    runs: list[AdaptiveCompressedRunStore] = []
    for run_id, block_id in enumerate((10, 11, 12)):
        keys, values = _full_state(2)
        runs.append(
            AdaptiveCompressedRunStore.from_full_block(
                run_id,
                block_id=block_id,
                keys=keys,
                values=values,
                group_size=32,
                bits=8,
            )
        )
    merged = runs[0].merge_with(runs[1], run_id=0).merge_with(runs[2], run_id=0)

    (q_keys, q_values), fragments = merged.split_without_block(
        11,
        left_run_id=20,
        right_run_id=21,
    )

    assert q_keys[0].shape[-2] == 2
    assert q_values[0].shape[-2] == 2
    assert len(fragments) == 2
    assert fragments[0].block_slices == ((10, 0, 2),)
    assert fragments[1].block_slices == ((12, 0, 2),)
    assert fragments[0].token_count == 2
    assert fragments[1].token_count == 2
