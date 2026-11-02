import random

import pytest

from turingserve.engine.sampling_params import SamplingParams
from turingserve.engine.sequence import Sequence
from turingserve.kv.block_allocator import BlockAllocator, NoFreeBlocks
from turingserve.kv.kv_manager import KVCacheManager
from turingserve.kv.prefix_cache import PrefixCache


def mk_seq(n_tokens: int, rid="r", base=0) -> Sequence:
    return Sequence(rid, [base + i for i in range(n_tokens)], n_tokens, SamplingParams(max_tokens=8))


def test_allocator_never_hands_out_null_block():
    a = BlockAllocator(8)
    got = [a.allocate() for _ in range(7)]
    assert sorted(got) == list(range(1, 8)) and 0 not in got
    with pytest.raises(NoFreeBlocks):
        a.allocate()


def test_allocator_random_invariants():
    rng = random.Random(0)
    a = BlockAllocator(50)
    owned: list[int] = []
    for _ in range(3000):
        op = rng.random()
        if op < 0.5:
            try:
                owned.append(a.allocate())
            except NoFreeBlocks:
                assert a.num_available() == 0
        elif op < 0.8 and owned:
            a.decref(owned.pop(rng.randrange(len(owned))))
        elif owned:
            b = rng.choice(owned)
            a.incref(b)
            owned.append(b)
        assert all(r >= 0 for r in a.ref)
        owned_ids = set(owned)
        assert not (owned_ids & set(a.free))
        assert len(set(a.free)) + len(owned_ids) + len(a.evictable) == 49
        assert 0 not in owned_ids and 0 not in a.free


@pytest.mark.parametrize("n,blocks", [(1, 1), (15, 1), (16, 1), (17, 2), (32, 2), (33, 3)])
def test_kv_manager_blocks_and_slots(n, blocks):
    kv = KVCacheManager(32)
    s = mk_seq(n)
    assert kv.blocks_needed(s, n) == blocks
    kv.allocate(s, n)
    assert len(s.block_table) == blocks
    for pos in range(n):
        assert kv.slot(s, pos) == s.block_table[pos // 16] * 16 + pos % 16
    kv.free(s)
    assert kv.allocator.num_available() == 31 and s.block_table == []


def test_trim_frees_trailing_blocks():
    kv = KVCacheManager(32)
    s = mk_seq(20)
    kv.allocate(s, 20 + 40)
    s.num_computed_tokens = 21
    kv.trim(s)
    assert len(s.block_table) == 2


def _run_prefill(kv, pc, s):
    pc.attach_hits(s)
    kv.allocate(s, s.num_uncomputed)
    s.num_computed_tokens = len(s.token_ids)
    pc.register_full_blocks(s)


def test_prefix_cache_same_prompt_twice():
    for L in (16, 17, 31, 32, 48, 50):
        kv = KVCacheManager(64)
        pc = PrefixCache(kv.allocator)
        a = mk_seq(L, "a")
        _run_prefill(kv, pc, a)
        kv.free(a)
        b = mk_seq(L, "b")
        pc.attach_hits(b)
        expect = 16 * (L // 16 - (1 if L % 16 == 0 else 0))
        assert b.num_computed_tokens == expect and b.num_cached_prompt_tokens == expect


def test_prefix_cache_shared_32_tokens():
    kv = KVCacheManager(64)
    pc = PrefixCache(kv.allocator)
    a = mk_seq(48, "a")
    _run_prefill(kv, pc, a)
    b = Sequence("b", list(range(32)) + [900 + i for i in range(20)], 52, SamplingParams())
    pc.attach_hits(b)
    assert len(b.block_table) == 2 and b.num_computed_tokens == 32


def test_eviction_only_ref0_and_hash_removed():
    kv = KVCacheManager(6)                       # 5 usable blocks
    pc = PrefixCache(kv.allocator)
    a = mk_seq(32, "a")
    _run_prefill(kv, pc, a)
    assert len(kv.allocator.hash_to_block) == 2
    # while a is running, its blocks are not evictable
    c = mk_seq(48, "c", base=500)
    kv.allocate(c, 48)
    with pytest.raises(NoFreeBlocks):
        kv.allocate(mk_seq(16, "d", base=900), 16)
    kv.free(a)                                    # blocks of a become evictable
    assert kv.allocator.num_cached == 2
    kv.allocate(mk_seq(32, "e", base=700), 32)    # forces eviction of both
    assert kv.allocator.hash_to_block == {} and kv.allocator.num_cached == 0


def test_attach_detach_roundtrip():
    kv = KVCacheManager(64)
    pc = PrefixCache(kv.allocator)
    a = mk_seq(40, "a")
    _run_prefill(kv, pc, a)
    kv.free(a)
    before = kv.allocator.num_available()
    b = mk_seq(40, "b")
    pc.attach_hits(b)
    pc.detach_hits(b)
    assert kv.allocator.num_available() == before and b.block_table == []
