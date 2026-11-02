"""Speculative-decoding acceptance rules (ARCH 6.13)."""

from __future__ import annotations

import torch


def accept_greedy(drafts: list[int], target_logits: torch.Tensor) -> tuple[int, int]:
    """target_logits [k+1, V]. Returns (m accepted drafts, next token: correction or bonus)."""
    argmax = target_logits.argmax(dim=-1).tolist()
    for i, d in enumerate(drafts):
        if d != argmax[i]:
            return i, argmax[i]
    return len(drafts), argmax[len(drafts)]


def _uniform(generator) -> float:
    if generator is None:
        return float(torch.rand(1).item())
    return float(torch.rand(1, generator=generator, device=generator.device).item())


def _sample(probs: torch.Tensor, generator) -> int:
    e = torch.empty_like(probs)
    if generator is None:
        e.exponential_()
    else:
        e.exponential_(generator=generator)
    return int((probs / e).argmax().item())


def accept_sampling(drafts: list[int], p: torch.Tensor, q: torch.Tensor | None,
                    generator=None) -> tuple[int, int]:
    """p [k+1, V] target probabilities, q [k, V] draft probabilities or None (n-gram point mass).

    Draft model: accept with prob min(1, p(d)/q(d)); on rejection sample normalize(max(0, p-q)).
    N-gram: accept with prob p(d); on rejection sample p with d removed and renormalized.
    All accepted -> sample the bonus from p[k]."""
    g = generator
    for i, d in enumerate(drafts):
        pi = p[i]
        if q is not None:
            qd = float(q[i, d])
            accept_prob = 1.0 if qd <= 0 else min(1.0, float(pi[d]) / qd)
        else:
            accept_prob = float(pi[d])
        if _uniform(g) < accept_prob:
            continue
        if q is not None:
            resid = torch.clamp(pi - q[i], min=0)
        else:
            resid = pi.clone()
            resid[d] = 0
        total = resid.sum()
        resid = resid / total if float(total) > 0 else pi
        return i, _sample(resid, g)
    return len(drafts), _sample(p[len(drafts)], g)
