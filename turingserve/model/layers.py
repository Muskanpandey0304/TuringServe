"""Weight containers and the rotary cache (ARCH 6.7)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


class RotaryCache:
    """cos/sin tables [max_len, D/2] in fp32 (cast to the model dtype at use, as HF does)."""

    def __init__(self, head_dim: int, theta: float, max_len: int, device) -> None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float()
                                    / head_dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
        self.cos = freqs.cos().to(device)
        self.sin = freqs.sin().to(device)


@dataclass
class LinearWeight:
    """A linear layer. ``w`` is [N, K] fp16/fp32, or int8 with per-row fp32 ``scale`` (W8A16)."""

    w: torch.Tensor
    b: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    @property
    def quantized(self) -> bool:
        return self.scale is not None


@dataclass
class LayerWeights:
    w_in: torch.Tensor
    qkv: LinearWeight
    o: LinearWeight
    w_post: torch.Tensor
    gu: LinearWeight
    down: LinearWeight


@dataclass
class Weights:
    embed: torch.Tensor                       # [V, hidden], also the lm head (tied)
    layers: list[LayerWeights] = field(default_factory=list)
    w_norm: torch.Tensor | None = None
    embed_q: tuple[torch.Tensor, torch.Tensor] | None = None   # optional INT8 lm-head copy
