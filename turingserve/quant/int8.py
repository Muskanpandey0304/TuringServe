"""INT8 weight-only quantization (W8A16), ARCH 6.14."""

from __future__ import annotations

import torch


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row symmetric quantization of W [N, K] -> (int8 [N, K], fp32 scale [N])."""
    scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-8).float()
    wq = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return wq, scale


def dequantize(wq: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (wq.to(torch.float32) * scale[:, None]).to(dtype)
