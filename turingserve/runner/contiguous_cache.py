"""Phase-2 contiguous KV cache. Kept as the test reference for the paged cache."""

from __future__ import annotations

import torch

from turingserve.model.qwen2_config import Qwen2Config


class ContiguousCache:
    def __init__(self, cfg: Qwen2Config, max_len: int, device, dtype) -> None:
        shape = (max_len, cfg.num_kv_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_layers)]

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        n = k.shape[0]
        self.k[layer][start:start + n] = k
        self.v[layer][start:start + n] = v

    def view(self, layer: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer][:length], self.v[layer][:length]
