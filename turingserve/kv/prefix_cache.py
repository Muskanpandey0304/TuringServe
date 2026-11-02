"""Hash-chain prefix cache (ARCH 6.6).

Only full, immutable blocks are shared, so no copy-on-write is needed.
"""

from __future__ import annotations

import hashlib
import struct

from turingserve.engine.sequence import Sequence
from turingserve.kv.block_allocator import BlockAllocator


def chain_hash(prev: bytes, tokens: list[int]) -> bytes:
    """h_i = blake2b(h_{i-1} + int32_bytes(tokens), digest_size=16); h_{-1} = b''."""
    return hashlib.blake2b(prev + struct.pack(f"<{len(tokens)}i", *tokens), digest_size=16).digest()


class PrefixCache:
    def __init__(self, allocator: BlockAllocator, block_size: int = 16, enabled: bool = True):
        self.alloc = allocator
        self.block_size = block_size
        self.enabled = enabled
        self.queried_tokens = 0
        self.hit_tokens = 0

    # -- registration (after every step) ---------------------------------------------------------
    def register_full_blocks(self, seq: Sequence) -> None:
        if not self.enabled:
            return
        bs = self.block_size
        full = min(seq.num_computed_tokens // bs, len(seq.block_table))
        while seq.num_hashed_blocks < full:
            i = seq.num_hashed_blocks
            prev = seq.hash_chain[i - 1] if i > 0 else b""
            h = chain_hash(prev, seq.token_ids[i * bs:(i + 1) * bs])
            if len(seq.hash_chain) > i:
                seq.hash_chain[i] = h
            else:
                seq.hash_chain.append(h)
            b = seq.block_table[i]
            # another block already owns this hash -> this block stays private
            if h not in self.alloc.hash_to_block and self.alloc.block_hash[b] is None:
                self.alloc.hash_to_block[h] = b
                self.alloc.block_hash[b] = h
            seq.num_hashed_blocks += 1

    # -- admission ---------------------------------------------------------------------------------
    def attach_hits(self, seq: Sequence) -> int:
        """Attach cached prefix blocks to a fresh sequence. Returns the number of hit tokens."""
        if not self.enabled or seq.num_computed_tokens != 0 or seq.block_table:
            return 0
        bs = self.block_size
        self.queried_tokens += len(seq.token_ids)
        prev = b""
        hits = 0
        for i in range(len(seq.token_ids) // bs):
            h = chain_hash(prev, seq.token_ids[i * bs:(i + 1) * bs])
            b = self.alloc.hash_to_block.get(h)
            if b is None:
                break
            self.alloc.incref(b)
            seq.block_table.append(b)
            seq.hash_chain.append(h)
            prev = h
            hits += 1
        if hits and hits * bs == len(seq.token_ids):
            # at least one token must be computed to produce logits: drop the last hit block
            self.alloc.decref(seq.block_table.pop())
            seq.hash_chain.pop()
            hits -= 1
        seq.num_computed_tokens = hits * bs
        seq.num_hashed_blocks = hits
        seq.num_cached_prompt_tokens = hits * bs
        self.hit_tokens += hits * bs
        return hits * bs

    def detach_hits(self, seq: Sequence) -> None:
        if not seq.block_table:
            return
        self.hit_tokens -= seq.num_cached_prompt_tokens
        self.queried_tokens -= len(seq.token_ids)
        for b in seq.block_table:
            self.alloc.decref(b)
        seq.block_table = []
        seq.hash_chain = []
        seq.num_computed_tokens = 0
        seq.num_hashed_blocks = 0
        seq.num_cached_prompt_tokens = 0

    @property
    def hit_rate(self) -> float:
        return self.hit_tokens / self.queried_tokens if self.queried_tokens else 0.0
