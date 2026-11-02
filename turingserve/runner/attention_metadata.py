"""AttentionMetadata and ModelInput (ARCH 5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionMetadata:
    query_start_loc: torch.Tensor     # [B+1] int32
    seq_lens: torch.Tensor            # [B]   int32, context length AFTER this step
    block_tables: torch.Tensor        # [B, max_blocks_per_seq] int32, padded with 0 (null block)
    slot_mapping: torch.Tensor        # [T] int64, -1 = padding
    num_decode: int                   # first num_decode rows have query_len 1 -> decode kernel
    max_query_len: int
    max_seq_len: int
    in_graph: bool = False            # True during CUDA-graph capture/replay (fixed grids, no host sync)


@dataclass
class ModelInput:
    input_ids: torch.Tensor           # [T] int64
    positions: torch.Tensor           # [T] int64
    meta: AttentionMetadata
    sample_indices: torch.Tensor      # [S] int64, rows of T whose hidden state is turned into logits
