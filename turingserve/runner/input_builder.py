"""Input builders (ARCH 6.10.1).

``build_simple`` is the Phase-3 Python version, kept as the test reference. ``InputBuilder`` is the
vectorized one with persistent (pinned, on CUDA) CPU buffers and persistent device buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from turingserve.engine.sequence import BatchPlan, Sequence
from turingserve.runner.attention_metadata import AttentionMetadata, ModelInput

BS = 16


@dataclass
class BatchItem:
    """``tokens`` are placed at positions start .. start+len-1 of ``seq``'s block table.
    sample: 'last' (one logits row), 'all' (a row per token; speculative verify) or 'none'."""

    seq: Sequence
    start: int
    tokens: list[int]
    sample: str = "last"


def items_from_plan(plan: BatchPlan) -> list[BatchItem]:
    out = []
    for it in plan.items:
        s = it.seq
        st = s.num_computed_tokens
        out.append(BatchItem(s, st, s.token_ids[st: st + it.num_tokens],
                             "last" if it.samples else "none"))
    return out


def _num_decode(items: list[BatchItem]) -> int:
    n = 0
    for it in items:
        if len(it.tokens) == 1 and it.sample != "all":
            n += 1
        else:
            break
    return n


def build_simple(items: list[BatchItem], device) -> ModelInput:
    ids, pos, slots, qsl, seq_lens, tables, sample = [], [], [], [0], [], [], []
    for it in items:
        n = len(it.tokens)
        p = list(range(it.start, it.start + n))
        bt = it.seq.block_table
        ids += it.tokens
        pos += p
        slots += [bt[x // BS] * BS + x % BS for x in p]
        qsl.append(qsl[-1] + n)
        seq_lens.append(it.start + n)
        tables.append(list(bt))
        if it.sample == "last":
            sample.append(qsl[-1] - 1)
        elif it.sample == "all":
            sample += list(range(qsl[-2], qsl[-1]))
    width = max(1, max(len(t) for t in tables))
    tables = [t + [0] * (width - len(t)) for t in tables]
    meta = AttentionMetadata(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        block_tables=torch.tensor(tables, dtype=torch.int32, device=device),
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
        num_decode=_num_decode(items), max_query_len=max(len(i.tokens) for i in items),
        max_seq_len=max(seq_lens))
    return ModelInput(torch.tensor(ids, dtype=torch.int64, device=device),
                      torch.tensor(pos, dtype=torch.int64, device=device), meta,
                      torch.tensor(sample, dtype=torch.int64, device=device))


class InputBuilder:
    def __init__(self, max_tokens: int, max_seqs: int, max_blocks_per_seq: int, device) -> None:
        self.device = torch.device(device)
        pin = self.device.type == "cuda"

        def host(shape, dtype):
            return torch.zeros(shape, dtype=dtype, pin_memory=pin)

        def dev(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        self.max_tokens, self.max_seqs, self.max_blocks = max_tokens, max_seqs, max_blocks_per_seq
        self.h_ids, self.d_ids = host(max_tokens, torch.int64), dev(max_tokens, torch.int64)
        self.h_pos, self.d_pos = host(max_tokens, torch.int64), dev(max_tokens, torch.int64)
        self.h_slot, self.d_slot = host(max_tokens, torch.int64), dev(max_tokens, torch.int64)
        self.h_sample, self.d_sample = host(max_tokens, torch.int64), dev(max_tokens, torch.int64)
        self.h_qsl, self.d_qsl = host(max_seqs + 1, torch.int32), dev(max_seqs + 1, torch.int32)
        self.h_sl, self.d_sl = host(max_seqs, torch.int32), dev(max_seqs, torch.int32)
        self.h_bt = host((max_seqs, max_blocks_per_seq), torch.int32)
        self.d_bt = dev((max_seqs, max_blocks_per_seq), torch.int32)
        self.np_ids, self.np_pos = self.h_ids.numpy(), self.h_pos.numpy()
        self.np_slot, self.np_sample = self.h_slot.numpy(), self.h_sample.numpy()
        self.np_qsl, self.np_sl, self.np_bt = self.h_qsl.numpy(), self.h_sl.numpy(), self.h_bt.numpy()

    def build(self, items: list[BatchItem]) -> ModelInput:
        """Pinned buffers are reused only after the step's sampler synchronization, so a copy
        from the previous step can never still be reading them."""
        B = len(items)
        if B > self.max_seqs:
            raise ValueError(f"batch of {B} sequences exceeds the builder's {self.max_seqs}")
        off = 0
        ns = 0
        width = max(1, max(len(it.seq.block_table) for it in items))
        self.np_qsl[0] = 0
        for i, it in enumerate(items):
            n = len(it.tokens)
            if off + n > self.max_tokens:
                raise ValueError("batch exceeds max_num_batched_tokens")
            pos = np.arange(it.start, it.start + n, dtype=np.int64)
            bt = np.asarray(it.seq.block_table, dtype=np.int64)
            self.np_ids[off:off + n] = it.tokens
            self.np_pos[off:off + n] = pos
            self.np_slot[off:off + n] = bt[pos // BS] * BS + pos % BS
            self.np_qsl[i + 1] = off + n
            self.np_sl[i] = it.start + n
            nb = len(bt)
            self.np_bt[i, :nb] = bt
            self.np_bt[i, nb:width] = 0            # rows are reused across steps: clear stale ids
            if it.sample == "last":
                self.np_sample[ns] = off + n - 1
                ns += 1
            elif it.sample == "all":
                self.np_sample[ns:ns + n] = np.arange(off, off + n)
                ns += n
            off += n
        T = off
        nb_ = False if self.device.type == "cpu" else True
        self.d_ids[:T].copy_(self.h_ids[:T], non_blocking=nb_)
        self.d_pos[:T].copy_(self.h_pos[:T], non_blocking=nb_)
        self.d_slot[:T].copy_(self.h_slot[:T], non_blocking=nb_)
        self.d_sample[:ns].copy_(self.h_sample[:ns], non_blocking=nb_)
        self.d_qsl[:B + 1].copy_(self.h_qsl[:B + 1], non_blocking=nb_)
        self.d_sl[:B].copy_(self.h_sl[:B], non_blocking=nb_)
        self.d_bt[:B, :width].copy_(self.h_bt[:B, :width], non_blocking=nb_)
        meta = AttentionMetadata(
            query_start_loc=self.d_qsl[:B + 1], seq_lens=self.d_sl[:B],
            block_tables=self.d_bt[:B, :width], slot_mapping=self.d_slot[:T],
            num_decode=_num_decode(items), max_query_len=max(len(i.tokens) for i in items),
            max_seq_len=int(self.np_sl[:B].max()))
        return ModelInput(self.d_ids[:T], self.d_pos[:T], meta, self.d_sample[:ns])
