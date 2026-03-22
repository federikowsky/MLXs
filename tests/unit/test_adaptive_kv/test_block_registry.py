from __future__ import annotations

from mlxs.adaptive_kv.block_registry import AdaptiveBlockRegistry
from mlxs.adaptive_kv.block_types import PinState


def test_prompt_block_creation_and_pin_state() -> None:
    registry = AdaptiveBlockRegistry(block_size_tokens=4)
    blocks = registry.initialize_prompt(10, step=0)

    assert [(block.start_token, block.end_token) for block in blocks] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert blocks[0].pin_state is PinState.HARD
    assert blocks[1].pin_state is PinState.NORMAL


def test_generated_blocks_extend_until_block_size_boundary() -> None:
    registry = AdaptiveBlockRegistry(block_size_tokens=3)
    registry.initialize_prompt(2, step=0)

    created = registry.ensure_generated_tokens(3, step=0)
    assert len(created) == 1
    block = registry.snapshot()[-1]
    assert (block.start_token, block.end_token) == (2, 3)

    created = registry.ensure_generated_tokens(5, step=1)
    assert created == []
    block = registry.snapshot()[-1]
    assert (block.start_token, block.end_token) == (2, 5)

    created = registry.ensure_generated_tokens(7, step=2)
    assert len(created) == 1
    assert (created[0].start_token, created[0].end_token) == (5, 7)


def test_token_slice_mapping_uses_existing_blocks() -> None:
    registry = AdaptiveBlockRegistry(block_size_tokens=2)
    registry.initialize_prompt(3, step=0)
    registry.ensure_generated_tokens(6, step=0)

    assert registry.token_slices(1, 5) == [
        (0, 0, 1),
        (1, 1, 2),
        (2, 2, 4),
    ]


def test_recent_tail_query() -> None:
    registry = AdaptiveBlockRegistry(block_size_tokens=2)
    registry.initialize_prompt(4, step=0)
    registry.ensure_generated_tokens(8, step=0)

    tail_ids = registry.recent_tail_block_ids(2)
    assert tail_ids == {2, 3}

