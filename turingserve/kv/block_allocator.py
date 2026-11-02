"""Block allocator (ARCH 6.5): pure Python, CPU-testable.

Block 0 is the null block and is never handed out. A block whose reference count drops to 0 goes
to ``evictable`` (LRU order) if it carries a prefix-cache hash, otherwise straight back to ``free``.
"""

from __future__ import annotations

from collections import OrderedDict, deque


class NoFreeBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 2:
            raise ValueError("need at least 2 blocks (block 0 is the null block)")
        self.num_blocks = num_blocks
        self.free: deque[int] = deque(range(1, num_blocks))
        self.ref: list[int] = [0] * num_blocks
        self.evictable: OrderedDict[int, None] = OrderedDict()
        # prefix cache bookkeeping lives here so that eviction can drop the mapping atomically
        self.block_hash: list[bytes | None] = [None] * num_blocks
        self.hash_to_block: dict[bytes, int] = {}

    def allocate(self) -> int:
        if self.free:
            b = self.free.popleft()
        elif self.evictable:
            b, _ = self.evictable.popitem(last=False)     # oldest first
            h = self.block_hash[b]
            if h is not None and self.hash_to_block.get(h) == b:
                del self.hash_to_block[h]
            self.block_hash[b] = None
        else:
            raise NoFreeBlocks()
        assert self.ref[b] == 0
        self.ref[b] = 1
        return b

    def incref(self, b: int) -> None:
        if self.ref[b] == 0:
            self.evictable.pop(b, None)
        self.ref[b] += 1

    def decref(self, b: int) -> None:
        assert self.ref[b] > 0, f"decref of unowned block {b}"
        self.ref[b] -= 1
        if self.ref[b] == 0:
            if self.block_hash[b] is not None:
                self.evictable[b] = None
            else:
                self.free.append(b)

    def num_available(self) -> int:
        return len(self.free) + len(self.evictable)

    @property
    def num_cached(self) -> int:
        return len(self.evictable)
