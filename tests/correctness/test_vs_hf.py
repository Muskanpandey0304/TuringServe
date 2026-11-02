"""Phases 1-2: logits, layers and greedy generation vs Hugging Face (tiny models, CPU, fp32)."""

import pytest
import torch

from tests.correctness.helpers import hf_greedy, load_hf, load_ours, rand_ids
from turingserve.model.qwen2_config import Qwen2Config
from turingserve.simple_generate import generate_greedy


@pytest.fixture(params=["g2", "g7"])
def tiny(request, tiny_model_dir, tiny_model_g7_dir):
    return tiny_model_dir if request.param == "g2" else tiny_model_g7_dir


def test_config_values(tiny):
    c = Qwen2Config.from_dir(tiny)
    assert c.group_size == c.num_heads // c.num_kv_heads
    assert c.tie_word_embeddings and c.eos_token_ids == (151645, 151643)


def test_config_rejects_unsupported():
    base = dict(model_type="qwen2", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=10,
                max_position_embeddings=64)
    Qwen2Config.from_dict(base)
    for bad in ({"model_type": "llama"}, {"use_sliding_window": True}, {"rope_scaling": {"a": 1}}):
        with pytest.raises(ValueError):
            Qwen2Config.from_dict({**base, **bad})


def test_loader_uses_every_tensor_and_fuses(tiny):
    m, c = load_ours(tiny)
    L = m.w.layers[0]
    assert L.qkv.w.shape == (c.qkv_dim, c.hidden_size) and L.qkv.b.shape == (c.qkv_dim,)
    assert L.gu.w.shape == (2 * c.intermediate_size, c.hidden_size)
    assert torch.isfinite(m.w.embed).all()


def test_logits_match_hf(tiny):
    ours, _ = load_ours(tiny)
    hf = load_hf(tiny)
    ids = rand_ids(37)
    with torch.inference_mode():
        ref = hf(torch.tensor([ids])).logits[0]
        h = ours.forward_dense(torch.tensor(ids), torch.arange(37))
        got = ours.compute_logits(h, torch.arange(37))
    assert (got - ref).abs().max() <= 1e-4


def test_layerwise_hidden_states_match_hf(tiny):
    ours, c = load_ours(tiny)
    hf = load_hf(tiny)
    ids = rand_ids(29, seed=1)
    with torch.inference_mode():
        ref = hf(torch.tensor([ids]), output_hidden_states=True).hidden_states
        _, got = ours.forward_dense(torch.tensor(ids), torch.arange(29), return_hidden=True)
    assert len(ref) == len(got) == c.num_layers + 1
    for i, (r, g) in enumerate(zip(ref, got)):
        assert (r[0] - g).abs().max() <= 1e-4, f"first diverging entry: {i}"


def test_cache_consistency_prefill_half_then_decode(tiny):
    ours, _ = load_ours(tiny)
    from turingserve.runner.contiguous_cache import ContiguousCache

    ids = rand_ids(40, seed=2)
    with torch.inference_mode():
        full = ours.compute_logits(ours.forward_dense(torch.tensor(ids), torch.arange(40)),
                                   torch.tensor([39]))
        cache = ContiguousCache(ours.cfg, 64, "cpu", torch.float32)
        ours.forward_contiguous(torch.tensor(ids[:20]), torch.arange(20), cache, 0)
        for i in range(20, 40):
            h = ours.forward_contiguous(torch.tensor([ids[i]]), torch.tensor([i]), cache, i)
        last = ours.compute_logits(h, torch.tensor([0]))
    assert (full - last).abs().max() <= 1e-4


def test_generate_matches_hf_greedy(tiny):
    ours, c = load_ours(tiny)
    hf = load_hf(tiny)
    for seed in range(3):
        prompt = rand_ids(11 + seed * 7, seed=10 + seed)
        want = hf_greedy(hf, prompt, 32, c.eos_token_ids)
        got = generate_greedy(ours, prompt, 32, c.eos_token_ids)
        assert got == want
