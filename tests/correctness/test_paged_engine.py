"""Phases 3-5, 8-9 on the tiny model (CPU, fp32): paged == contiguous, batch invariance,
engine == HF greedy, input-builder equivalence, prefix cache and preemption with a real model."""

import random

import pytest
import torch

from tests.correctness.helpers import hf_greedy, load_hf, load_ours, rand_ids
from turingserve.config import EngineConfig
from turingserve.engine.engine import LLMEngine
from turingserve.engine.output_processor import OutputProcessor
from turingserve.engine.sampling_params import SamplingParams
from turingserve.engine.sequence import Sequence
from turingserve.kv.kv_manager import KVCacheManager
from turingserve.kv.prefix_cache import PrefixCache
from turingserve.runner.input_builder import BatchItem, InputBuilder, build_simple
from turingserve.runner.model_runner import ModelRunner
from turingserve.sampling.sampler import Sampler
from turingserve.scheduler.scheduler import Scheduler


def make_engine(path, num_blocks=300, budget=64, max_seqs=16, prefix=True, shuffle=False, tok=None):
    model, mcfg = load_ours(path)
    cfg = EngineConfig(model=path, device="cpu", kernels="reference", dtype="float32",
                       max_model_len=2048, max_num_batched_tokens=budget, max_num_seqs=max_seqs,
                       num_blocks=num_blocks, autopilot="off")
    kv_cache = torch.zeros(mcfg.num_layers, 2, num_blocks, mcfg.num_kv_heads, 16, mcfg.head_dim)
    kv = KVCacheManager(num_blocks)
    if shuffle:                                   # non-contiguous block ids
        ids = list(kv.allocator.free)
        random.Random(0).shuffle(ids)
        kv.allocator.free.clear()
        kv.allocator.free.extend(ids)
    pc = PrefixCache(kv.allocator, enabled=prefix)
    sch = Scheduler(kv, pc, budget, max_seqs, 0.0)
    runner = ModelRunner(model, kv_cache, budget, max_seqs, cfg.max_blocks_per_seq, "cpu")
    op = OutputProcessor(tok, mcfg.eos_token_ids, 2048)
    return LLMEngine(cfg, runner, kv, pc, sch, Sampler(mcfg.eos_token_ids), op,
                     mcfg.eos_token_ids), model, mcfg


GREEDY = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)  # noqa: E731


def test_paged_logits_equal_dense(tiny_model_dir):
    eng, model, mcfg = make_engine(tiny_model_dir)
    ids = rand_ids(70, seed=5)
    seq = Sequence("a", ids, 70, GREEDY(1))
    eng.kv.allocate(seq, 70)
    inp = build_simple([BatchItem(seq, 0, ids, "all")], "cpu")
    with torch.inference_mode():
        got = model.compute_logits(model.forward(inp.input_ids, inp.positions, inp.meta,
                                                 eng.runner.kv_cache), inp.sample_indices)
        from turingserve.runner.contiguous_cache import ContiguousCache

        cache = ContiguousCache(mcfg, 128, "cpu", torch.float32)
        ref = model.compute_logits(
            model.forward_contiguous(torch.tensor(ids), torch.arange(70), cache, 0), torch.arange(70))
        dense = model.compute_logits(model.forward_dense(torch.tensor(ids), torch.arange(70)),
                                     torch.arange(70))
    assert (got - ref).abs().max() <= 1e-5          # paged == contiguous
    assert (got - dense).abs().max() <= 1e-4


def test_batch_invariance_and_shuffled_blocks(tiny_model_dir):
    lengths = [5, 16, 17, 100, 333, 512, 1000, 2047]
    prompts = [rand_ids(n, seed=n) for n in lengths]
    alone = []
    for p in prompts:
        eng, model, _ = make_engine(tiny_model_dir, num_blocks=200, shuffle=True)
        seq = Sequence("a", p, len(p), GREEDY(1))
        eng.kv.allocate(seq, len(p))
        inp = build_simple([BatchItem(seq, 0, p, "all")], "cpu")
        with torch.inference_mode():
            alone.append(model.compute_logits(model.forward(
                inp.input_ids, inp.positions, inp.meta, eng.runner.kv_cache), inp.sample_indices))
    eng, model, _ = make_engine(tiny_model_dir, num_blocks=600, shuffle=True)
    items = []
    for i, p in enumerate(prompts):
        s = Sequence(str(i), p, len(p), GREEDY(1))
        eng.kv.allocate(s, len(p))
        items.append(BatchItem(s, 0, p, "all"))
    inp = build_simple(items, "cpu")
    with torch.inference_mode():
        together = model.compute_logits(model.forward(
            inp.input_ids, inp.positions, inp.meta, eng.runner.kv_cache), inp.sample_indices)
    off = 0
    for n, a in zip(lengths, alone, strict=False):
        assert (together[off:off + n] - a).abs().max() <= 1e-4, n
        off += n


def test_engine_greedy_equals_hf_with_chunking_and_batching(tiny_model_dir, tokenizer):
    hf = load_hf(tiny_model_dir)
    prompts = [rand_ids(n, seed=n) for n in (7, 40, 130, 33, 90)]
    eng, _, mcfg = make_engine(tiny_model_dir, budget=48, tok=tokenizer)
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=24))
    for p, o in zip(prompts, outs, strict=False):
        assert o.token_ids == hf_greedy(hf, p, 24, mcfg.eos_token_ids)[: len(o.token_ids)]
        assert len(o.token_ids) == 24 or o.finish_reason == "stop"


def test_preemption_with_real_model_same_outputs(tiny_model_dir):
    prompts = [rand_ids(40 + i, seed=i) for i in range(6)]
    small, *_ = make_engine(tiny_model_dir, num_blocks=16, budget=32)
    big, *_ = make_engine(tiny_model_dir, num_blocks=400, budget=32)
    a = small.generate(prompts, GREEDY(30))
    b = big.generate(prompts, GREEDY(30))
    assert small.scheduler.num_preemptions > 0
    assert [x.token_ids for x in a] == [x.token_ids for x in b]


def test_prefix_cache_with_real_model_same_outputs(tiny_model_dir):
    shared = rand_ids(100, seed=1)
    prompts = [shared + rand_ids(5, seed=10 + i) for i in range(4)]
    on, *_ = make_engine(tiny_model_dir, prefix=True)
    off, *_ = make_engine(tiny_model_dir, prefix=False)
    r_on = [on.generate([p], GREEDY(10))[0] for p in prompts]
    r_off = [off.generate([p], GREEDY(10))[0] for p in prompts]
    assert [x.token_ids for x in r_on] == [x.token_ids for x in r_off]
    assert r_on[1].cached_prompt_tokens == 96


def test_vectorized_input_builder_equals_python_builder():
    rng = random.Random(0)
    builder = InputBuilder(2048, 16, 128, "cpu")
    kv = KVCacheManager(400)
    for _ in range(200):
        items = []
        for i in range(rng.randint(1, 8)):
            n_prompt = rng.randint(1, 120)
            s = Sequence(str(i), [rng.randrange(1000) for _ in range(n_prompt)], n_prompt,
                         GREEDY(4))
            start = rng.randint(0, n_prompt - 1)
            n = rng.randint(1, n_prompt - start)
            s.num_computed_tokens = start
            kv.allocate(s, n)
            items.append(BatchItem(s, start, s.token_ids[start:start + n],
                                   rng.choice(["last", "none", "all"])))
        items.sort(key=lambda it: len(it.tokens) != 1)
        a, b = build_simple(items, "cpu"), builder.build(items)
        assert torch.equal(a.input_ids, b.input_ids) and torch.equal(a.positions, b.positions)
        assert torch.equal(a.meta.slot_mapping, b.meta.slot_mapping)
        assert torch.equal(a.meta.query_start_loc, b.meta.query_start_loc)
        assert torch.equal(a.meta.seq_lens, b.meta.seq_lens)
        w = a.meta.block_tables.shape[1]
        assert torch.equal(a.meta.block_tables, b.meta.block_tables[:, :w])
        assert torch.equal(a.sample_indices, b.sample_indices)
        assert a.meta.num_decode == b.meta.num_decode
        for it in items:
            kv.free(it.seq)


def test_seeded_sampling_reproducible_through_engine(tiny_model_dir):
    p = rand_ids(20, seed=3)
    outs = []
    for _ in range(2):
        eng, *_ = make_engine(tiny_model_dir)
        extra = [rand_ids(10, seed=i) for i in range(5)]
        sp = SamplingParams(temperature=1.0, seed=42, max_tokens=16, ignore_eos=True)
        o = eng.generate([p] + extra, [sp] + [SamplingParams(max_tokens=16, ignore_eos=True)] * 5)
        outs.append(o[0].token_ids)
    assert outs[0] == outs[1]


@pytest.mark.parametrize("variant", ["g2", "g7"])
def test_speculative_ngram_real_model_matches_plain(variant, tiny_model_dir, tiny_model_g7_dir):
    from turingserve.spec.ngram import NgramProposer
    from turingserve.spec.spec_decoder import SpecDecoder

    path = tiny_model_dir if variant == "g2" else tiny_model_g7_dir
    prompt = rand_ids(8, seed=9) * 6                           # repetitive -> n-gram hits
    plain, *_ = make_engine(path)
    spec, *_ = make_engine(path)
    spec.spec = SpecDecoder(spec, NgramProposer(3, 1), 4, 4, 8, "ngram")
    a = plain.generate([prompt], GREEDY(30))[0]
    b = spec.generate([prompt], GREEDY(30))[0]
    # fp32: only a numerical near-tie could differ; the random tiny model has spread logits
    assert a.token_ids == b.token_ids
