"""Adaptive block registry."""

from __future__ import annotations

from dataclasses import replace

from mlxs.adaptive_kv.block_types import BlockRecord, PinState, ResidentProfile


class AdaptiveBlockRegistry:
    """Tracks logical adaptive blocks in chronological order."""

    def __init__(self, block_size_tokens: int) -> None:
        self._block_size_tokens = block_size_tokens
        self._blocks: list[BlockRecord] = []
        self._next_block_id = 0
        self._prompt_token_count = 0

    @property
    def blocks(self) -> tuple[BlockRecord, ...]:
        return tuple(self._blocks)

    @property
    def total_tokens(self) -> int:
        if not self._blocks:
            return 0
        return self._blocks[-1].end_token

    def initialize_prompt(self, prompt_token_count: int, *, step: int) -> list[BlockRecord]:
        if self._blocks:
            raise ValueError("Prompt blocks already initialized")
        self._prompt_token_count = prompt_token_count
        created: list[BlockRecord] = []
        for start in range(0, prompt_token_count, self._block_size_tokens):
            end = min(prompt_token_count, start + self._block_size_tokens)
            pin_state = PinState.HARD if start == 0 else PinState.NORMAL
            structural_prior = 1.0 if pin_state is PinState.HARD else 0.2
            created.append(
                self._append_block(
                    start_token=start,
                    end_token=end,
                    source_start=start,
                    source_end=end,
                    segment_id=0,
                    pin_state=pin_state,
                    structural_prior=structural_prior,
                    step=step,
                )
            )
        return created

    def ensure_generated_tokens(self, total_tokens: int, *, step: int) -> list[BlockRecord]:
        """Create generated blocks so the registry covers ``total_tokens``."""
        created: list[BlockRecord] = []
        if total_tokens <= self._prompt_token_count:
            return created
        covered = self.total_tokens
        while covered < total_tokens:
            tail = self._blocks[-1] if self._blocks else None
            if (
                tail is not None
                and tail.segment_id == 1
                and tail.token_count < self._block_size_tokens
            ):
                new_end = min(total_tokens, tail.start_token + self._block_size_tokens)
                extended = replace(
                    tail,
                    end_token=new_end,
                    source_end=new_end,
                )
                self._blocks[-1] = extended
                covered = extended.end_token
                continue
            start = covered
            end = min(total_tokens, start + self._block_size_tokens)
            created.append(
                self._append_block(
                    start_token=start,
                    end_token=end,
                    source_start=start,
                    source_end=end,
                    segment_id=1,
                    pin_state=PinState.NORMAL,
                    structural_prior=0.0,
                    step=step,
                )
            )
            covered = end
        return created

    def active_blocks(self) -> list[BlockRecord]:
        return [block for block in self._blocks if block.profile is not ResidentProfile.EVICTED]

    def resident_blocks(self) -> list[BlockRecord]:
        return [block for block in self._blocks if block.profile is not ResidentProfile.EVICTED]

    def covered_blocks(self, total_tokens: int) -> list[BlockRecord]:
        """Blocks whose spans begin before ``total_tokens``."""
        return [block for block in self._blocks if block.start_token < total_tokens]

    def covered_resident_blocks(self, total_tokens: int) -> list[BlockRecord]:
        """Resident blocks whose spans begin before ``total_tokens``."""
        return [
            block
            for block in self._blocks
            if block.profile is not ResidentProfile.EVICTED and block.start_token < total_tokens
        ]

    def evicted_blocks(self) -> list[BlockRecord]:
        return [block for block in self._blocks if block.profile is ResidentProfile.EVICTED]

    def required_evicted_blocks(
        self,
        total_tokens: int,
        *,
        start_token: int = 0,
    ) -> list[BlockRecord]:
        """Evicted blocks whose spans intersect the required prefix window."""
        return [
            block
            for block in self._blocks
            if (
                block.profile is ResidentProfile.EVICTED
                and block.start_token < total_tokens
                and block.end_token > start_token
            )
        ]

    def get(self, block_id: int) -> BlockRecord:
        for block in self._blocks:
            if block.block_id == block_id:
                return block
        raise KeyError(block_id)

    def update(self, block: BlockRecord) -> None:
        for idx, current in enumerate(self._blocks):
            if current.block_id == block.block_id:
                self._blocks[idx] = block
                return
        raise KeyError(block.block_id)

    def mark_profile(self, block_id: int, profile: ResidentProfile) -> BlockRecord:
        block = self.get(block_id)
        block = replace(block, profile=profile)
        self.update(block)
        return block

    def block_for_token(self, token_index: int) -> BlockRecord:
        for block in self._blocks:
            if block.start_token <= token_index < block.end_token:
                return block
        raise KeyError(token_index)

    def token_slices(self, start_token: int, end_token: int) -> list[tuple[int, int, int]]:
        """Return block ids and local token slices for ``[start_token, end_token)``."""
        slices: list[tuple[int, int, int]] = []
        cursor = start_token
        while cursor < end_token:
            block = self.block_for_token(cursor)
            local_start = cursor - start_token
            local_end = min(end_token, block.end_token) - start_token
            slices.append((block.block_id, local_start, local_end))
            cursor = block.end_token
        return slices

    def recent_tail_block_ids(self, n_blocks: int) -> set[int]:
        if n_blocks <= 0:
            return set()
        return {block.block_id for block in self._blocks[-n_blocks:]}

    def snapshot(self) -> list[BlockRecord]:
        return list(self._blocks)

    def _append_block(
        self,
        *,
        start_token: int,
        end_token: int,
        source_start: int,
        source_end: int,
        segment_id: int,
        pin_state: PinState,
        structural_prior: float,
        step: int,
    ) -> BlockRecord:
        block = BlockRecord(
            block_id=self._next_block_id,
            start_token=start_token,
            end_token=end_token,
            source_start=source_start,
            source_end=source_end,
            segment_id=segment_id,
            pin_state=pin_state,
            profile=ResidentProfile.TQ_SAFE,
            created_step=step,
            structural_prior=structural_prior,
        )
        self._next_block_id += 1
        self._blocks.append(block)
        return block
