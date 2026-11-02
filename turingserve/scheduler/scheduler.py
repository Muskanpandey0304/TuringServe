"""Scheduler (ARCH 6.4): continuous batching + chunked prefill + recompute preemption."""

from __future__ import annotations

from collections import deque

from turingserve.engine.sequence import BatchPlan, Scheduled, SeqStatus, Sequence
from turingserve.kv.kv_manager import KVCacheManager
from turingserve.kv.prefix_cache import PrefixCache


class Scheduler:
    def __init__(self, kv: KVCacheManager, prefix_cache: PrefixCache | None,
                 max_num_batched_tokens: int, max_num_seqs: int, watermark: float = 0.01):
        self.kv = kv
        self.prefix_cache = prefix_cache
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.watermark_blocks = max(1, int(watermark * kv.num_blocks))
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    # -- queue management ------------------------------------------------------------------------
    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def finish(self, seq: Sequence) -> None:
        """Remove a finished/aborted sequence from the queues and free its blocks."""
        if seq in self.running:
            self.running.remove(seq)
        else:
            try:
                self.waiting.remove(seq)
            except ValueError:
                pass
        self.kv.free(seq)

    def _preempt(self, seq: Sequence) -> None:
        """Recompute-style: generated tokens stay in token_ids, KV is dropped."""
        self.kv.free(seq)
        seq.num_computed_tokens = 0
        seq.block_table = []
        seq.num_hashed_blocks = 0
        seq.hash_chain = []
        seq.draft_computed = 0
        seq.num_cached_prompt_tokens = 0
        seq.status = SeqStatus.WAITING
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def preempt_all(self) -> list[Sequence]:
        """Used on CUDA OOM (ARCH 6.17)."""
        victims = list(self.running)
        for s in reversed(victims):
            self._preempt(s)
        return victims

    # -- the schedule -----------------------------------------------------------------------------
    def schedule(self) -> BatchPlan:
        budget = self.max_num_batched_tokens
        plan: list[Scheduled] = []
        in_plan: set[int] = set()
        preempted: list[Sequence] = []
        preempted_ids: set[int] = set()

        # Pass 1: decoding sequences, oldest first
        for seq in [s for s in self.running if s.num_uncomputed == 1]:
            if id(seq) in preempted_ids or budget <= 0:
                continue
            while not self.kv.can_append(seq, 1):
                victim = None
                for cand in reversed(self.running):
                    if cand is not seq and id(cand) not in in_plan:
                        victim = cand
                        break
                if victim is None:
                    victim = seq
                self._preempt(victim)
                preempted.append(victim)
                preempted_ids.add(id(victim))
                if victim is seq:
                    break
            if id(seq) in preempted_ids:
                continue
            self.kv.allocate(seq, 1)
            plan.append(Scheduled(seq, 1, True))
            in_plan.add(id(seq))
            budget -= 1

        # Pass 2: running sequences in the middle of a chunked prefill (never preempt for these)
        for seq in [s for s in self.running if s.num_uncomputed > 1 and id(s) not in preempted_ids]:
            if budget == 0:
                break
            n = min(seq.num_uncomputed, budget)
            if not self.kv.can_append(seq, n):
                break
            self.kv.allocate(seq, n)
            plan.append(Scheduled(seq, n, n == seq.num_uncomputed))
            in_plan.add(id(seq))
            budget -= n

        # Pass 3: admit waiting sequences FCFS, only if nothing was preempted in this step
        while (self.waiting and budget > 0 and not preempted
               and len(self.running) < self.max_num_seqs):
            seq = self.waiting[0]
            if self.prefix_cache is not None and seq.num_computed_tokens == 0:
                self.prefix_cache.attach_hits(seq)
            n = min(seq.num_uncomputed, budget)
            if not self.kv.can_append(seq, n, keep_free=self.watermark_blocks):
                if self.prefix_cache is not None:
                    self.prefix_cache.detach_hits(seq)
                break
            self.waiting.popleft()
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            self.kv.allocate(seq, n)
            plan.append(Scheduled(seq, n, n == seq.num_uncomputed))
            budget -= n

        decodes = [p for p in plan if p.num_tokens == 1 and p.seq.num_uncomputed == 1]
        prefills = [p for p in plan if not (p.num_tokens == 1 and p.seq.num_uncomputed == 1)]
        items = decodes + prefills
        return BatchPlan(items=items, num_decode=len(decodes),
                         num_tokens=sum(p.num_tokens for p in items), preempted=preempted)
