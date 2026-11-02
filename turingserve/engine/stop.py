"""Finish checks after each new token (ARCH 6.12), in the documented order."""

from __future__ import annotations

from turingserve.engine.sequence import Sequence


def is_stop_token(seq: Sequence, token: int, eos_ids: tuple[int, ...]) -> bool:
    if token in seq.params.stop_token_ids:
        return True
    return (not seq.params.ignore_eos) and token in eos_ids


def find_stop_string(seq: Sequence) -> int:
    """Index in ``seq.text`` of the earliest new stop-string match, or -1."""
    best = -1
    for s in seq.params.stop:
        k = seq.text.find(s, max(0, seq.emitted_len - len(s) + 1))
        if k != -1 and (best == -1 or k < best):
            best = k
    return best


def hold_back(seq: Sequence) -> int:
    n = seq.params.max_stop_len
    return n - 1 if n > 0 else 0
