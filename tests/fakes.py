"""Test doubles. FakeModelRunner gives EXACT, deterministic outputs for scheduler / preemption /
prefix-cache / speculation tests: next token = (sum(last 3 token ids) + position) % 1000, where
position is the index of the row's last input token."""

from __future__ import annotations

import torch

from turingserve.config import EngineConfig
from turingserve.engine.engine import LLMEngine
from turingserve.engine.output_processor import OutputProcessor
from turingserve.kv.kv_manager import KVCacheManager
from turingserve.kv.prefix_cache import PrefixCache
from turingserve.sampling.sampler import Sampler
from turingserve.scheduler.scheduler import Scheduler

V = 1000
EOS = (999,)


def next_token(ctx: list[int]) -> int:
    """ctx = all token ids up to and including the row's last input token."""
    return (sum(ctx[-3:]) + (len(ctx) - 1)) % 1000


def onehot(tok: int) -> torch.Tensor:
    row = torch.full((V,), -10.0)
    row[tok] = 10.0
    return row


class FakeModelRunner:
    """Same interface as ModelRunner. Also records every plan for invariant checks."""

    def __init__(self, engine_ref=None):
        self.plans = []
        self.graphs = None

    def execute(self, plan):
        self.plans.append(plan)
        rows = []
        for it in plan.items:
            if it.samples:
                s = it.seq
                ctx = s.token_ids[: s.num_computed_tokens + it.num_tokens]
                rows.append(onehot(next_token(ctx)))
        return torch.stack(rows) if rows else None

    def execute_verify(self, entries):
        out = []
        for seq, toks in entries:
            base = seq.token_ids[: seq.num_computed_tokens]
            rows = []
            ctx = list(base)
            for t in toks:
                ctx.append(t)
                rows.append(onehot(next_token(ctx)))
            out.append(torch.stack(rows))
        return out


def make_fake_engine(num_blocks=200, budget=64, max_num_seqs=32, prefix_caching=True,
                     max_model_len=4096, watermark=0.0, spec=None) -> LLMEngine:
    cfg = EngineConfig(model="fake", device="cpu", kernels="reference", dtype="float32",
                       max_model_len=max_model_len, max_num_batched_tokens=budget,
                       max_num_seqs=max_num_seqs, prefix_caching=prefix_caching,
                       num_blocks=num_blocks, autopilot="off", watermark=watermark)
    kv = KVCacheManager(num_blocks)
    pc = PrefixCache(kv.allocator, enabled=prefix_caching)
    sch = Scheduler(kv, pc, budget, max_num_seqs, watermark)
    sampler = Sampler(EOS, 0, "cpu")
    op = OutputProcessor(None, EOS, max_model_len)
    return LLMEngine(cfg, FakeModelRunner(), kv, pc, sch, sampler, op, EOS, device="cpu")


def expected_output(prompt: list[int], n: int) -> list[int]:
    """What the fake model generates greedily (no EOS before n tokens is assumed by callers)."""
    ctx = list(prompt)
    out = []
    for _ in range(n):
        t = next_token(ctx)
        out.append(t)
        ctx.append(t)
    return out
