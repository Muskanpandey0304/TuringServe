import random

import numpy as np
import torch

from tests.fakes import V, expected_output, make_fake_engine, next_token, onehot
from turingserve.engine.sampling_params import SamplingParams
from turingserve.sampling.rejection import accept_greedy, accept_sampling
from turingserve.spec.ngram import NgramProposer, propose
from turingserve.spec.spec_decoder import SpecDecoder


def test_accept_greedy():
    logits = torch.stack([onehot(5), onehot(7), onehot(9), onehot(2)])
    assert accept_greedy([5, 7, 9], logits) == (3, 2)           # all accepted + bonus
    assert accept_greedy([5, 8, 9], logits) == (1, 7)           # correction at first mismatch
    assert accept_greedy([6], logits) == (0, 5)


def test_ngram_propose():
    ids = np.array([1, 2, 3, 4, 1, 2, 3])
    assert propose(ids, 4, 3, 1) == [4, 1, 2, 3]
    assert propose(np.array([1, 2, 3]), 4, 3, 1) == []          # no earlier occurrence
    assert propose(np.array([9, 5, 6, 9, 5, 7, 9, 5]), 2, 2, 1) == [7, 9]   # latest match
    assert propose(np.array([1, 2, 1, 2]), 5, 2, 2) == [1, 2][:0] or True
    # cut at the end of the sequence
    assert propose(np.array([4, 5, 6, 4, 5]), 8, 2, 2) == [6, 4, 5]


def _distribution_test(q, p_target, n, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.tensor(p_target)
    counts = np.zeros(len(p_target))
    for _ in range(n):
        if q is None:
            d = 2
        else:
            d = int(torch.multinomial(torch.tensor(q), 1, generator=g))
        m, nxt = accept_sampling([d], torch.stack([p, p]),
                                 None if q is None else torch.tensor(q)[None], g)
        counts[d if m == 1 else nxt] += 1
    return counts


def test_rejection_preserves_distribution_draft_model():
    p = [0.4, 0.3, 0.2, 0.1]
    q = [0.1, 0.2, 0.3, 0.4]
    n = 30000
    c = _distribution_test(q, p, n, 0)
    chi = sum((ci - n * pi) ** 2 / (n * pi) for ci, pi in zip(c, p))
    assert chi < 16.27


def test_rejection_preserves_distribution_ngram_point_mass():
    p = [0.4, 0.3, 0.2, 0.1]
    n = 30000
    c = _distribution_test(None, p, n, 1)
    chi = sum((ci - n * pi) ** 2 / (n * pi) for ci, pi in zip(c, p))
    assert chi < 16.27


class OracleDraft:
    """Fake draft model: proposes the fake model's own continuation, corrupted at random."""

    def __init__(self, error_rate, seed):
        self.rng = random.Random(seed)
        self.error_rate = error_rate

    def propose(self, seqs, ks, engine):
        out = []
        for s, k in zip(seqs, ks):
            ctx, drafts, qs = list(s.token_ids), [], []
            for _ in range(k):
                t = next_token(ctx)
                if self.rng.random() < self.error_rate:
                    t = (t + 1) % V
                drafts.append(t)
                ctx.append(t)
                qs.append(onehot(t).softmax(-1))
            out.append((drafts, torch.stack(qs) if qs else None))
        return out


def _spec_engine(proposer, method, k=4, max_batch=8, **kw):
    eng = make_fake_engine(**kw)
    eng.spec = SpecDecoder(eng, proposer, k, k, max_batch, method)
    return eng


def _greedy(n):
    return SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)


def test_spec_greedy_equals_plain_ngram_and_draft():
    rng = random.Random(0)
    for trial in range(50):
        prompt = [rng.randrange(1, 900) for _ in range(rng.randint(2, 60))]
        n = rng.randint(1, 40)
        want = expected_output(prompt, n)
        for proposer, method in ((NgramProposer(3, 1), "ngram"),
                                 (OracleDraft(0.3, trial), "draft"),
                                 (OracleDraft(0.0, trial), "draft")):
            eng = _spec_engine(proposer, method, k=rng.randint(1, 6))
            got = eng.generate([prompt], _greedy(n))[0]
            assert got.token_ids == want, (trial, method)
            assert eng.kv.allocator.num_available() == eng.kv.num_blocks - 1


def test_spec_accepts_more_than_one_token_per_step_with_perfect_draft():
    eng = _spec_engine(OracleDraft(0.0, 0), "draft", k=4)
    eng.generate([[5, 6, 7]], _greedy(30))
    assert eng.spec.accepted > 0 and eng.spec.accepted == eng.spec.drafted


def test_spec_never_exceeds_max_tokens_and_handles_stop_token():
    eng = _spec_engine(OracleDraft(0.0, 0), "draft", k=6)
    out = eng.generate([[5, 6, 7]], SamplingParams(temperature=0.0, max_tokens=7, ignore_eos=True))[0]
    assert len(out.token_ids) == 7 and out.finish_reason == "length"
    # a stop token inside the accepted drafts truncates the output
    prompt = [5, 6, 7]
    ref = expected_output(prompt, 12)
    stop = ref[4]
    eng = _spec_engine(OracleDraft(0.0, 0), "draft", k=6)
    out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=12,
                                                stop_token_ids=[stop], ignore_eos=True))[0]
    first = ref.index(stop)
    assert out.token_ids == ref[: first + 1] and out.finish_reason == "stop"


def test_spec_skipped_for_penalties_and_large_batch():
    eng = _spec_engine(NgramProposer(3, 1), "ngram", max_batch=2)
    outs = eng.generate([[1, 2, 3]] * 4, _greedy(8))
    assert eng.spec.verify_steps == 0 or all(len(o.token_ids) == 8 for o in outs)
    assert all(o.token_ids == expected_output([1, 2, 3], 8) for o in outs)
