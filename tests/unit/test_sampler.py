import math

import numpy as np
import torch

from turingserve.engine.sampling_params import SamplingParams
from turingserve.engine.sequence import Sequence
from turingserve.sampling.penalties import apply_penalties
from turingserve.sampling.sampler import Sampler

EOS = (3,)


def seq(params, seed=None, prompt=(1, 2), out=()):
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    ids = list(prompt) + list(out)
    return Sequence("r", ids, len(prompt), params, generator=g)


def chi2(counts, probs, n):
    return sum((c - n * p) ** 2 / (n * p) for c, p in zip(counts, probs) if p > 0)


def draw(sampler, logits_row, params, n):
    logits = logits_row[None].repeat(n, 1)
    seqs = [seq(params) for _ in range(n)]
    toks, bad = sampler.sample(logits, seqs)
    return np.bincount(toks, minlength=logits_row.numel()), bad


def test_greedy_equivalences():
    s = Sampler(EOS)
    logits = torch.randn(5, 50)
    greedy, _ = s.sample(logits, [seq(SamplingParams(temperature=0.0)) for _ in range(5)])
    topk1, _ = s.sample(logits, [seq(SamplingParams(temperature=1.0, top_k=1)) for _ in range(5)])
    assert (greedy == logits.argmax(-1).numpy()).all() and (topk1 == greedy).all()


def test_seed_reproducible_alone_and_in_batch():
    s = Sampler(EOS)
    row = torch.randn(200)
    p = SamplingParams(temperature=1.0, seed=7)
    alone, _ = s.sample(row[None], [seq(p, seed=7)])
    batch = [seq(SamplingParams(temperature=1.0), seed=None) for _ in range(7)]
    mine = seq(p, seed=7)
    logits = torch.cat([torch.randn(7, 200), row[None]])
    together, _ = s.sample(logits, batch + [mine])
    assert alone[0] == together[-1]


def test_distribution_chi_square():
    s = Sampler(EOS, seed=1)
    probs = [0.5, 0.3, 0.15, 0.05]
    row = torch.tensor(probs).log()
    n = 20000
    counts, _ = draw(s, row, SamplingParams(temperature=1.0), n)
    assert chi2(counts, probs, n) < 16.27


def test_top_k_2():
    s = Sampler(EOS, seed=2)
    row = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
    n = 20000
    counts, _ = draw(s, row, SamplingParams(temperature=1.0, top_k=2), n)
    assert counts[2] == 0 and counts[3] == 0
    assert chi2(counts[:2], [0.625, 0.375], n) < 10.83


def test_top_p_075_keeps_two_tokens():
    s = Sampler(EOS, seed=3)
    row = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
    counts, _ = draw(s, row, SamplingParams(temperature=1.0, top_p=0.75), 5000)
    assert counts[2] == 0 and counts[3] == 0 and counts[0] > 0 and counts[1] > 0


def test_temperature_half_squares_probabilities():
    s = Sampler(EOS, seed=4)
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = p**2 / (p**2).sum()
    n = 20000
    counts, _ = draw(s, torch.tensor(p).float().log(), SamplingParams(temperature=0.5), n)
    assert chi2(counts, q, n) < 16.27


def test_penalties_values():
    logits = torch.tensor([[4.0, -4.0, 1.0, 0.0]])
    out = apply_penalties(logits, [[0, 1]], [[]], torch.tensor([2.0]), torch.tensor([0.0]),
                          torch.tensor([0.0]))
    assert out[0, 0] == 2.0 and out[0, 1] == -8.0 and out[0, 2] == 1.0
    out = apply_penalties(torch.zeros(1, 4), [[]], [[2, 2]], torch.tensor([1.0]),
                          torch.tensor([0.0]), torch.tensor([0.5]))
    assert math.isclose(out[0, 2].item(), -1.0)
    out = apply_penalties(torch.zeros(1, 4), [[]], [[1]], torch.tensor([1.0]),
                          torch.tensor([0.3]), torch.tensor([0.0]))
    assert math.isclose(out[0, 1].item(), -0.3, rel_tol=1e-5)


def test_min_tokens_masks_eos():
    s = Sampler(EOS)
    logits = torch.zeros(1, 10)
    logits[0, 3] = 100.0
    p = SamplingParams(temperature=0.0, min_tokens=2, max_tokens=5)
    tok, _ = s.sample(logits, [seq(p)])
    assert tok[0] != 3
    sq = seq(p, out=(7, 8))                    # two output tokens exist -> EOS allowed again
    tok, _ = s.sample(logits, [sq])
    assert tok[0] == 3


def test_nan_row_flagged_bad():
    s = Sampler(EOS)
    logits = torch.randn(2, 10)
    logits[1, 4] = float("nan")
    tok, bad = s.sample(logits, [seq(SamplingParams(temperature=0.0)) for _ in range(2)])
    assert list(bad) == [False, True]
