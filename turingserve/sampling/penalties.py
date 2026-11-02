"""Repetition (HF semantics) and presence/frequency (OpenAI semantics) penalties (ARCH 6.11)."""

from __future__ import annotations

import torch


def _padded(rows: list[list[int]], V: int, device) -> torch.Tensor:
    width = max(1, max((len(r) for r in rows), default=1))
    arr = torch.full((len(rows), width), V, dtype=torch.int64)      # V = padding column
    for i, r in enumerate(rows):
        if r:
            arr[i, : len(r)] = torch.tensor(r, dtype=torch.int64)
    return arr.to(device)


def apply_penalties(logits: torch.Tensor, prompt_ids: list[list[int]], output_ids: list[list[int]],
                    repetition: torch.Tensor, presence: torch.Tensor,
                    frequency: torch.Tensor) -> torch.Tensor:
    """logits [S, V] fp32; penalty tensors [S]. Returns new logits."""
    S, V = logits.shape
    dev = logits.device
    out_idx = _padded(output_ids, V, dev)
    counts_out = torch.zeros(S, V + 1, dtype=torch.float32, device=dev)
    counts_out.scatter_add_(1, out_idx, torch.ones_like(out_idx, dtype=torch.float32))
    counts_out = counts_out[:, :V]
    prompt_idx = _padded(prompt_ids, V, dev)
    seen = torch.zeros(S, V + 1, dtype=torch.bool, device=dev)
    seen.scatter_(1, prompt_idx, True)
    seen.scatter_(1, out_idx, True)
    seen = seen[:, :V]
    rp = repetition[:, None].to(dev)
    penalized = torch.where(logits > 0, logits / rp, logits * rp)
    logits = torch.where(seen, penalized, logits)
    logits = logits - frequency[:, None].to(dev) * counts_out
    logits = logits - presence[:, None].to(dev) * (counts_out > 0).float()
    return logits
