"""CUDA-graph replay for pure-decode steps (ARCH 6.10.2).

Forbidden during capture: .item(), .cpu(), print(tensor), tensors built from Python data,
Triton autotuning and any host-device synchronization. The kernel tuner must have finished.
"""

from __future__ import annotations

import torch

from turingserve.runner.attention_metadata import AttentionMetadata, ModelInput
from turingserve.utils.logging import log


class CudaGraphRunner:
    def __init__(self, model, kv_cache: torch.Tensor, sizes: tuple[int, ...],
                 max_blocks_per_seq: int, max_model_len: int, device) -> None:
        self.model, self.kv_cache, self.device = model, kv_cache, device
        self.sizes = sorted(set(sizes))
        self.max_b = self.sizes[-1]
        self.max_model_len = max_model_len
        V = model.cfg.vocab_size
        i32, i64 = torch.int32, torch.int64
        self.ids = torch.zeros(self.max_b, dtype=i64, device=device)
        self.pos = torch.zeros(self.max_b, dtype=i64, device=device)
        self.slot = torch.full((self.max_b,), -1, dtype=i64, device=device)
        self.seq_lens = torch.zeros(self.max_b, dtype=i32, device=device)
        self.bt = torch.zeros(self.max_b, max_blocks_per_seq, dtype=i32, device=device)
        self.qsl = torch.arange(self.max_b + 1, dtype=i32, device=device)
        self.sample_idx = torch.arange(self.max_b, dtype=i64, device=device)
        self.logits = torch.zeros(self.max_b, V, dtype=torch.float32, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def _meta(self, B: int) -> AttentionMetadata:
        return AttentionMetadata(
            query_start_loc=self.qsl[:B + 1], seq_lens=self.seq_lens[:B],
            block_tables=self.bt[:B], slot_mapping=self.slot[:B], num_decode=B,
            max_query_len=1, max_seq_len=self.max_model_len, in_graph=True)

    def _forward(self, B: int) -> None:
        hidden = self.model.forward(self.ids[:B], self.pos[:B], self._meta(B), self.kv_cache)
        self.logits[:B] = self.model.compute_logits(hidden, self.sample_idx[:B])

    def capture(self) -> None:
        # warm-up: every size eagerly, twice (compiles the Triton kernels)
        for B in self.sizes:
            for _ in range(2):
                self._forward(B)
        torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle()
        for B in reversed(self.sizes):               # largest first so smaller graphs share the pool
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._forward(B)
            self.graphs[B] = g
        torch.cuda.synchronize()
        log("cuda_graphs_captured", sizes=self.sizes)

    def can_replay(self, n: int) -> bool:
        return 0 < n <= self.max_b and bool(self.graphs)

    def replay(self, inp: ModelInput, n: int) -> torch.Tensor:
        """Copy the n real decode rows into the static buffers, pad up to the graph size."""
        B = next(s for s in self.sizes if s >= n)
        self.ids[:n].copy_(inp.input_ids[:n])
        self.pos[:n].copy_(inp.positions[:n])
        self.slot[:n].copy_(inp.meta.slot_mapping[:n])
        self.seq_lens[:n].copy_(inp.meta.seq_lens[:n])
        self.bt[:B].zero_()
        w = inp.meta.block_tables.shape[1]
        self.bt[:n, :w].copy_(inp.meta.block_tables[:n])
        if B > n:
            self.ids[n:B].zero_()
            self.pos[n:B].zero_()
            self.slot[n:B].fill_(-1)
            self.seq_lens[n:B].zero_()
        self.graphs[B].replay()
        return self.logits[:n]
