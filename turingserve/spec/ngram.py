"""Prompt-lookup (n-gram) proposer (ARCH 6.13)."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def propose(token_ids: np.ndarray, k: int, n_max: int, n_min: int) -> list[int]:
    """For n from n_max down to n_min: find the LATEST earlier occurrence of the last n tokens and
    propose the (up to k) tokens that followed it. No match -> []."""
    ids = np.asarray(token_ids)
    L = len(ids)
    for n in range(n_max, n_min - 1, -1):
        if L <= n or k <= 0:
            continue
        pat = ids[-n:]
        # windows that start before the final suffix itself, so a match is an EARLIER occurrence
        windows = sliding_window_view(ids[: L - 1], n)
        hits = np.nonzero((windows == pat).all(axis=1))[0]
        if len(hits):
            i = int(hits[-1])
            cont = ids[i + n: i + n + k]
            if len(cont):
                return cont.tolist()
    return []


class NgramProposer:
    """Proposes from each sequence's own token history. q is a point mass (returned as None)."""

    def __init__(self, n_max: int, n_min: int) -> None:
        self.n_max, self.n_min = n_max, n_min

    def propose(self, seqs, ks: list[int], engine=None):
        out = []
        for seq, k in zip(seqs, ks, strict=False):
            out.append((propose(np.asarray(seq.token_ids), k, self.n_max, self.n_min), None))
        return out
