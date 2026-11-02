# KV cache and prefix cache (turingserve/kv/)
One tensor [L, 2, num_blocks, KVH, 16, D]; block 0 is the null block (padding rows point at it, slot -1 means skip).
Per sequence: a block table; slot(pos) = table[pos // 16] * 16 + pos % 16.
Allocator: free deque + ref counts + an LRU `evictable` map for ref-0 blocks that carry a hash. Prefix cache: chain
hash h_i = blake2b(h_{i-1} + tokens of block i); only full, computed blocks are registered; a fully cached prompt
gives back its last block so at least one token is computed (logits need it). Only immutable blocks are shared,
so there is no copy-on-write.
