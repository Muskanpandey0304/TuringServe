import json
import os

import torch

from turingserve.autopilot import tuner
from turingserve.autopilot.controller import Limits, Signals, State, decide
from turingserve.quant.int8 import dequantize, quantize

LIM = Limits(ttft_target_s=2.0, tpot_target_s=0.05, max_chunk=2048, max_num_seqs=128, spec_k_max=4,
             spec_max_batch=8, spec_enabled=True)


def st(**kw):
    return State(chunk=kw.pop("chunk", 2048), max_num_seqs=kw.pop("max_num_seqs", 64),
                 spec_k=kw.pop("spec_k", 4), **kw)


def run(state, sigs, dt=2.0, start=100.0, lim=LIM):
    now, decisions = start, []
    for s in sigs:
        state, d = decide(state, s, now, lim)
        decisions.append(d)
        now += dt
    return state, decisions


# -- quantization -------------------------------------------------------------------------------
def test_quantize_error_within_half_scale():
    w = torch.randn(64, 128) * 3
    wq, scale = quantize(w)
    assert wq.dtype == torch.int8 and wq.abs().max() <= 127
    assert ((dequantize(wq, scale, torch.float32) - w).abs() <= scale[:, None] / 2 + 1e-6).all()


# -- controller -----------------------------------------------------------------------------------
def test_stable_signals_no_decisions():
    _, d = run(st(), [Signals(ttft_p90=0.5, tpot_p90=0.03, kv_usage=0.8)] * 10)
    assert all(not x for x in d)


def test_violation_needs_three_windows():
    bad = Signals(tpot_p90=0.08, kv_usage=0.8)
    _, d = run(st(), [bad, bad])
    assert not any(d)
    s, d = run(st(), [bad, bad, bad])
    assert [len(x) for x in d] == [0, 0, 1] and d[2][0].rule == "R1" and s.chunk == 1024


def test_cooldown_respected():
    bad = Signals(tpot_p90=0.08, kv_usage=0.8)
    s, d = run(st(), [bad] * 3)
    assert s.chunk == 1024
    s, d = run(s, [bad] * 3, start=106.0)            # 3 more windows but inside the 10 s cooldown
    assert not any(d) and s.chunk == 1024
    s, d = run(s, [bad] * 3, start=130.0)
    assert s.chunk == 512


def test_bounds_never_crossed():
    bad = Signals(tpot_p90=0.08, kv_usage=0.8)
    s, _ = run(st(chunk=256), [bad] * 12, dt=11.0)
    assert s.chunk == 256
    hi = Signals(ttft_p90=5.0, tpot_p90=0.01, kv_usage=0.8)
    s, _ = run(st(chunk=2048), [hi] * 12, dt=11.0)
    assert s.chunk == 2048                           # capped by the flag (max_chunk)


def test_r2_fires_with_headroom_but_not_in_hysteresis_band():
    s, d = run(st(chunk=512), [Signals(ttft_p90=5.0, tpot_p90=0.02, kv_usage=0.8)] * 3)
    assert s.chunk == 1024 and d[2][0].rule == "R2"
    s, d = run(st(chunk=512), [Signals(ttft_p90=5.0, tpot_p90=0.04, kv_usage=0.8)] * 6)
    assert s.chunk == 512 and not any(d)             # 0.8 x target: inside the band


def test_r3_decode_batch_over_spec_max_batch_is_immediate():
    _, d = run(st(), [Signals(decode_batch=9, kv_usage=0.8)])
    assert d[0] and d[0][0].rule == "R3"


def test_r4_r5_spec_k_and_off_then_reprobe():
    low = Signals(accepted_ema=0.3, kv_usage=0.8)
    s, d = run(st(spec_k=3), [low] * 3)
    assert s.spec_k == 2 and d[2][0].rule == "R4"
    s, d = run(st(spec_k=1), [low] * 3, start=100.0)
    assert not s.spec_on and s.spec_off_until == 104.0 + 60.0
    s2, d2 = decide(s, Signals(kv_usage=0.8), 170.0, LIM)    # >= 60 s later: re-probe with k = 2
    assert s2.spec_on and s2.spec_k == 2
    high = Signals(accepted_ema=3.5, kv_usage=0.8)
    s, d = run(st(spec_k=3), [high] * 3)
    assert s.spec_k == 4 and d[2][0].rule == "R5"
    s, d = run(st(spec_k=4), [high] * 6, dt=11.0)
    assert s.spec_k == 4


def test_r6_r7_concurrency():
    s, d = run(st(max_num_seqs=64), [Signals(preemptions=9, kv_usage=0.95)] * 3)
    assert s.max_num_seqs == 48 and d[2][0].rule == "R6"
    s, d = run(st(max_num_seqs=8), [Signals(preemptions=9, kv_usage=0.95)] * 9, dt=11.0)
    assert s.max_num_seqs == 8
    s, d = run(st(max_num_seqs=64), [Signals(preemptions=0, kv_usage=0.3)] * 6)
    assert s.max_num_seqs == 80 and d[5][0].rule == "R7"
    s, _ = run(st(max_num_seqs=128), [Signals(preemptions=0, kv_usage=0.3)] * 12, dt=11.0)
    assert s.max_num_seqs == 128


def test_r0_oom_halves_immediately_never_below_256():
    s, d = run(st(chunk=2048), [Signals(oom=True)])
    assert s.chunk == 1024 and d[0][0].rule == "R0"
    s, _ = run(st(chunk=300), [Signals(oom=True)] * 3, dt=1.0)
    assert s.chunk == 256


# -- tuner cache ------------------------------------------------------------------------------------
def test_cache_key_contains_every_field():
    k = tuner.cache_key("Tesla T4", "qwen", "int8", 4096, 2048, "2.6.0+cu124", "3.2.0")
    for part in ("Tesla", "T4", "qwen", "int8", "mml4096", "tok2048", "torch2.6.0", "triton3.2.0"):
        assert part in k


def test_cache_roundtrip_and_corruption(tmp_path):
    t = tuner.TuningTable()
    t.set("prefill", "64", {"block_m": 64, "block_n": 32, "num_warps": 4, "num_stages": 2})
    t.m_cross["896,896"] = 128
    path = tuner.save_cache(str(tmp_path), "k.json", t)
    back = tuner.load_cache(str(tmp_path), "k.json")
    assert back.tables == t.tables and back.m_cross == t.m_cross
    open(path, "w").write("{not json")
    assert tuner.load_cache(str(tmp_path), "k.json") is None
    assert tuner.load_cache(str(tmp_path), "missing.json") is None
    open(path, "w").write(json.dumps({"nope": 1}))
    assert tuner.load_cache(str(tmp_path), "k.json") is None
    assert os.path.exists(path)


def test_lookup_rounds_up_and_falls_back():
    t = tuner.TuningTable()
    t.set("decode_v2", "4,512", {"block_n": 16, "num_warps": 2})
    t.set("decode_v2", "8,1024", {"block_n": 64, "num_warps": 4})
    assert t.lookup_rounded("decode_v2", (3, 300), {})["block_n"] == 16
    assert t.lookup_rounded("decode_v2", (5, 300), {})["block_n"] == 64
    assert t.lookup_rounded("prefill", (10,), {"x": 1}) == {"x": 1}
