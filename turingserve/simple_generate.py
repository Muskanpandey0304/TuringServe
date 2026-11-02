"""Phase-2 helper: greedy generation with a contiguous KV cache."""

from __future__ import annotations

import torch

from turingserve.runner.contiguous_cache import ContiguousCache


@torch.inference_mode()
def generate_greedy(model, prompt_ids: list[int], max_new_tokens: int,
                    eos_ids: tuple[int, ...] = ()) -> list[int]:
    dev = model.device
    cache = ContiguousCache(model.cfg, len(prompt_ids) + max_new_tokens, dev, model.dtype)
    ids = torch.tensor(prompt_ids, dtype=torch.int64, device=dev)
    pos = torch.arange(len(prompt_ids), dtype=torch.int64, device=dev)
    hidden = model.forward_contiguous(ids, pos, cache, 0)
    last = torch.tensor([len(prompt_ids) - 1], device=dev)
    out: list[int] = []
    length = len(prompt_ids)
    for _ in range(max_new_tokens):
        tok = int(model.compute_logits(hidden, last).argmax(-1).item())
        out.append(tok)
        if tok in eos_ids:
            break
        hidden = model.forward_contiguous(
            torch.tensor([tok], dtype=torch.int64, device=dev),
            torch.tensor([length], dtype=torch.int64, device=dev), cache, length)
        last = torch.tensor([0], device=dev)
        length += 1
    return out
