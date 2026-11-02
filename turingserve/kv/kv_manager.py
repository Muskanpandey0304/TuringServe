"""KV cache manager (ARCH 6.5): block tables and slot arithmetic."""

from __future__ import annotations

from turingserve.engine.sequence import Sequence
from turingserve.kv.block_allocator import BlockAllocator

BLOCK_SIZE = 16


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int = BLOCK_SIZE) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.num_blocks = num_blocks

    def blocks_needed(self, seq: Sequence, n: int) -> int:
        total = -(-(seq.num_computed_tokens + n) // self.block_size)
        return max(0, total - len(seq.block_table))

    def can_append(self, seq: Sequence, n: int, keep_free: int = 0) -> bool:
        return self.allocator.num_available() - self.blocks_needed(seq, n) >= keep_free

    def allocate(self, seq: Sequence, n: int) -> None:
        for _ in range(self.blocks_needed(seq, n)):
            seq.block_table.append(self.allocator.allocate())

    def free(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.allocator.decref(b)
        seq.block_table = []

    def trim(self, seq: Sequence) -> None:
        """Free trailing blocks that hold no computed token (after speculative verification)."""
        keep = -(-seq.num_computed_tokens // self.block_size)
        while len(seq.block_table) > keep:
            self.allocator.decref(seq.block_table.pop())

    def slot(self, seq: Sequence, pos: int) -> int:
        return seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size

    # -- observability -------------------------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.allocator.free)

    @property
    def num_cached(self) -> int:
        return self.allocator.num_cached

    @property
    def usage(self) -> float:
        """Fraction of blocks owned by running sequences."""
        owned = self.num_blocks - 1 - self.allocator.num_available()
        return owned / max(1, self.num_blocks - 1)
