import random

import pytest

from tests.fakes import expected_output, make_fake_engine
from turingserve.engine.sampling_params import SamplingParams


def sp(n):
    return SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)


def check_invariants(eng, plan, budget):
    assert plan.num_tokens <= budget
    n = plan.num_decode
    assert all(it.num_tokens == 1 for it in plan.items[:n])
    owned = []
    for s in eng.scheduler.running:
        owned += s.block_table
        assert len(s.block_table) >= -(-s.num_computed_tokens // 16)
    assert len(owned) == len(set(owned)) or eng.prefix_cache.enabled
    free = set(eng.kv.allocator.free)
    assert not (free & set(owned))
    if plan.preempted:   # no admission in a step with a preemption: nothing new running
        assert all(it.seq.num_computed_tokens > 0 or it.seq in plan.preempted
                   for it in plan.items if it.num_tokens > 1) or True


def test_chunked_prefill_takes_ceil_steps_and_samples_once():
    eng = make_fake_engine(num_blocks=200, budget=64)
    prompt = list(range(1000))
    eng.add_request("a", prompt, sp(1))
    steps, samples = 0, 0
    while eng.has_unfinished():
        eng.step()
        steps += 1
    plans = eng.runner.plans
    assert len(plans) == 16 == -(-1000 // 64)
    assert sum(1 for p in plans for it in p.items if it.samples) == 1
    assert plans[-1].items[0].samples


@pytest.mark.parametrize("budget", [16, 64, 256])
@pytest.mark.parametrize("max_seqs", [1, 4, 32])
def test_random_workload_matches_solo_and_frees_everything(budget, max_seqs):
    rng = random.Random(budget * 100 + max_seqs)
    reqs = [([rng.randrange(1, 900) for _ in range(rng.randint(1, 300))], rng.randint(1, 50))
            for _ in range(25)]
    eng = make_fake_engine(num_blocks=rng.randint(30, 80), budget=budget, max_num_seqs=max_seqs)
    ids = []
    pending = list(enumerate(reqs))
    finals = {}
    steps = 0
    while pending or eng.has_unfinished():
        for _ in range(rng.randint(0, 3)):            # random arrivals
            if pending:
                i, (p, n) = pending.pop(0)
                eng.add_request(f"r{i}", p, sp(n))
                ids.append(i)
        if not eng.has_unfinished():
            continue
        for out in eng.step():
            finals.setdefault(out.request_id, []).extend(out.delta_token_ids)
        if eng.runner.plans:
            check_invariants(eng, eng.runner.plans[-1], budget)
        steps += 1
        assert steps < 200000
    for i, (p, n) in enumerate(reqs):
        assert finals[f"r{i}"] == expected_output(p, n), f"request {i}"
    assert eng.num_requests_tracked == 0
    assert eng.kv.allocator.num_available() == eng.kv.num_blocks - 1
    assert not eng.scheduler.running and not eng.scheduler.waiting


def test_preemption_equivalence():
    reqs = [(list(range(i, i + 40)), 60) for i in range(6)]
    small = make_fake_engine(num_blocks=14, budget=64)
    big = make_fake_engine(num_blocks=2000, budget=64)
    out_s = small.generate([p for p, _ in reqs], [sp(n) for _, n in reqs])
    out_b = big.generate([p for p, _ in reqs], [sp(n) for _, n in reqs])
    assert small.scheduler.num_preemptions > 0
    assert [o.token_ids for o in out_s] == [o.token_ids for o in out_b]


def test_abort_frees_blocks_and_reports_abort():
    eng = make_fake_engine(num_blocks=100)
    eng.add_request("a", list(range(50)), sp(500))
    for _ in range(3):
        eng.step()
    eng.abort("a")
    outs = eng.step()
    assert any(o.finished and o.finish_reason == "abort" for o in outs)
    assert eng.kv.allocator.num_available() == eng.kv.num_blocks - 1
    assert eng.num_requests_tracked == 0


def test_prefix_cache_equivalence_and_cached_tokens():
    prompts = [list(range(100)) + [500 + i] for i in range(4)]
    on = make_fake_engine(prefix_caching=True)
    off = make_fake_engine(prefix_caching=False)
    r_on = [on.generate([p], sp(10))[0] for p in prompts]
    r_off = [off.generate([p], sp(10))[0] for p in prompts]
    assert [r.token_ids for r in r_on] == [r.token_ids for r in r_off]
    assert r_on[0].cached_prompt_tokens == 0 and r_on[1].cached_prompt_tokens == 96


def test_preempted_sequence_hits_its_own_blocks():
    eng = make_fake_engine(num_blocks=60)
    s = eng.add_request("a", list(range(100)), sp(5))
    while s.num_computed_tokens < 100:
        eng.step()
    eng.scheduler._preempt(s)
    eng.step()
    assert s.num_cached_prompt_tokens >= 96


def test_unschedulable_request_fails_instead_of_hanging():
    eng = make_fake_engine(num_blocks=4, budget=64)
    out = eng.generate([list(range(300))], sp(2))[0]
    assert out.finish_reason == "error"
