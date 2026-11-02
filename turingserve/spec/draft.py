"""Draft-model proposer (ARCH 6.13): a second Qwen2 (the 0.5B) with its OWN KV tensor of the same
``num_blocks``, sharing the target's block tables (same token positions), so it needs no allocator."""

from __future__ import annotations

import torch

from turingserve.runner.input_builder import BatchItem
from turingserve.sampling.rejection import _sample


class DraftProposer:
    def __init__(self, runner) -> None:
        self.runner = runner          # a ModelRunner over the draft model and draft KV tensor
        self.cap = runner.builder.max_tokens

    def _run_groups(self, items: list[BatchItem]) -> list[torch.Tensor]:
        """Run items (all sample='last') in groups that fit the builder; one logits row per item."""
        rows: list[torch.Tensor] = []
        i = 0
        while i < len(items):
            group, tokens = [], 0
            while i < len(items) and len(group) < self.runner.builder.max_seqs:
                n = len(items[i].tokens)
                if n > self.cap:               # a very long catch-up: feed it in chunks
                    break
                if tokens + n > self.cap:
                    break
                group.append(items[i])
                tokens += n
                i += 1
            if group:
                logits = self.runner.run_items(group)
                rows += [logits[j] for j in range(len(group))]
            elif i < len(items):
                it = items[i]
                st = it.start
                for c in range(0, len(it.tokens), self.cap):
                    chunk = it.tokens[c: c + self.cap]
                    last = c + self.cap >= len(it.tokens)
                    logits = self.runner.run_items(
                        [BatchItem(it.seq, st + c, chunk, "last" if last else "none")])
                    if last:
                        rows.append(logits[0])
                i += 1
        return rows

    def propose(self, seqs, ks: list[int], engine):
        sampler = engine.sampler
        L0 = [len(s.token_ids) for s in seqs]
        # 1. catch-up + first proposal
        items = [BatchItem(s, s.draft_computed, s.token_ids[s.draft_computed:], "last") for s in seqs]
        rows = self._run_groups(items)
        drafts: list[list[int]] = [[] for _ in seqs]
        qs: list[list[torch.Tensor]] = [[] for _ in seqs]

        def pick(i, row, j):
            s = seqs[i]
            q = sampler.processed_probs(row, s, num_output=s.num_output + j)
            tok = int(q.argmax()) if s.params.greedy else _sample(q, s.generator)
            drafts[i].append(tok)
            qs[i].append(q)

        for i, s in enumerate(seqs):
            s.draft_computed = L0[i]
            if ks[i] >= 1:
                pick(i, rows[i], 0)
        # 2. j = 2..k: feed d_{j-1} through the draft (decode path)
        for j in range(2, max(ks, default=0) + 1):
            live = [i for i in range(len(seqs)) if ks[i] >= j]
            items = [BatchItem(seqs[i], L0[i] + j - 2, [drafts[i][-1]], "last") for i in live]
            rows = self._run_groups(items)
            for r, i in zip(rows, live, strict=False):
                pick(i, r, j - 1)
                seqs[i].draft_computed = L0[i] + j - 1
        return [(drafts[i], torch.stack(qs[i]) if qs[i] else None) for i in range(len(seqs))]
