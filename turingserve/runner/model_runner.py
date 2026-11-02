"""Model runner: builds inputs, runs the forward pass, returns logits for sampling rows."""

from __future__ import annotations

import torch

from turingserve.engine.sequence import BatchPlan, Sequence
from turingserve.runner.input_builder import BatchItem, InputBuilder, items_from_plan


class ModelRunner:
    def __init__(self, model, kv_cache: torch.Tensor, max_num_batched_tokens: int, max_num_seqs: int,
                 max_blocks_per_seq: int, device, graph_runner=None) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.device = device
        self.graphs = graph_runner
        self.builder = InputBuilder(max_num_batched_tokens, max_num_seqs, max_blocks_per_seq, device)

    def run_items(self, items: list[BatchItem]) -> torch.Tensor | None:
        inp = self.builder.build(items)
        n_dec = inp.meta.num_decode
        if (self.graphs is not None and n_dec == len(items) and self.graphs.can_replay(n_dec)):
            return self.graphs.replay(inp, n_dec)
        hidden = self.model.forward(inp.input_ids, inp.positions, inp.meta, self.kv_cache)
        if inp.sample_indices.numel() == 0:
            return None
        return self.model.compute_logits(hidden, inp.sample_indices)

    def execute(self, plan: BatchPlan) -> torch.Tensor | None:
        """Logits [S, V] fp32 for the plan's sampling items (in plan order); None if none sample."""
        return self.run_items(items_from_plan(plan))

    def execute_verify(self, entries: list[tuple[Sequence, list[int]]]) -> list[torch.Tensor]:
        """Speculative verification: each entry is (seq, [last_uncomputed_token, d1..dk]).
        Returns one [1+k, V] logits tensor per entry."""
        items = [BatchItem(s, s.num_computed_tokens, toks, "all") for s, toks in entries]
        logits = self.run_items(items)
        out, row = [], 0
        for _, toks in entries:
            out.append(logits[row: row + len(toks)])
            row += len(toks)
        return out
