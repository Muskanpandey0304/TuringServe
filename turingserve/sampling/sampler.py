"""Sampler (ARCH 6.11): penalties -> min_tokens -> greedy / temperature -> top-k -> top-p ->
exponential-race sampling. One GPU->CPU copy at the end."""

from __future__ import annotations

import numpy as np
import torch

from turingserve.engine.sequence import Sequence
from turingserve.sampling.penalties import apply_penalties

NEG_INF = float("-inf")


class Sampler:
    def __init__(self, eos_ids: tuple[int, ...], seed: int = 0, device="cpu") -> None:
        self.eos_ids = list(eos_ids)
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    # -- shared pieces (also used by speculative verification) ----------------------------------
    def mask_stop_tokens(self, logits_row: torch.Tensor, seq: Sequence) -> torch.Tensor:
        ids = self.eos_ids + list(seq.params.stop_token_ids)
        logits_row = logits_row.clone()
        logits_row[ids] = NEG_INF
        return logits_row

    def processed_probs(self, logits_row: torch.Tensor, seq: Sequence, num_output: int | None = None
                        ) -> torch.Tensor:
        """Distribution after min_tokens / temperature / top-k / top-p for one row [V]."""
        p = seq.params
        n_out = seq.num_output if num_output is None else num_output
        row = logits_row.float()
        if n_out < p.min_tokens:
            row = self.mask_stop_tokens(row, seq)
        if p.greedy:
            probs = torch.zeros_like(row)
            probs[int(row.argmax())] = 1.0
            return probs
        probs = self._filtered_probs(row[None], torch.tensor([p.temperature]),
                                     torch.tensor([p.top_k]), torch.tensor([p.top_p]))[0]
        return probs

    @staticmethod
    def _filtered_probs(logits: torch.Tensor, temp: torch.Tensor, top_k: torch.Tensor,
                        top_p: torch.Tensor) -> torch.Tensor:
        """Probabilities in the ORIGINAL vocabulary order after temperature, top-k, top-p."""
        dev = logits.device
        S, V = logits.shape
        scaled = logits / temp[:, None].to(dev)
        if bool((top_k == -1).all()) and bool((top_p >= 1.0).all()):
            return torch.softmax(scaled, dim=-1)
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        pos = torch.arange(V, device=dev)[None, :]
        k = torch.where(top_k < 0, torch.full_like(top_k, V), top_k).to(dev)[:, None]
        sorted_logits = sorted_logits.masked_fill(pos >= k, NEG_INF)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_before = probs.cumsum(-1) - probs
        drop = (cum_before >= top_p[:, None].to(dev)) & (pos > 0)
        sorted_logits = sorted_logits.masked_fill(drop, NEG_INF)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        out = torch.zeros_like(sorted_probs)
        out.scatter_(1, sorted_idx, sorted_probs)
        return out

    # -- main entry --------------------------------------------------------------------------------
    @torch.inference_mode()
    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> tuple[np.ndarray, np.ndarray]:
        """logits [S, V] fp32 (one row per sequence in ``seqs``). Returns (tokens[S], bad[S])."""
        S, V = logits.shape
        dev = logits.device
        bad = ~torch.isfinite(logits).all(dim=-1)
        logits = torch.where(bad[:, None], torch.zeros_like(logits), logits)   # never sample NaN

        if any(s.params.uses_penalties for s in seqs):
            logits = apply_penalties(
                logits, [s.token_ids[: s.num_prompt_tokens] for s in seqs],
                [s.output_ids for s in seqs],
                torch.tensor([s.params.repetition_penalty for s in seqs]),
                torch.tensor([s.params.presence_penalty for s in seqs]),
                torch.tensor([s.params.frequency_penalty for s in seqs]))
        for i, s in enumerate(seqs):
            if s.num_output < s.params.min_tokens:
                logits[i] = self.mask_stop_tokens(logits[i], s)

        greedy = torch.tensor([s.params.greedy for s in seqs], device=dev)
        if bool(greedy.all()):
            tokens = logits.argmax(dim=-1)
        else:
            temp = torch.tensor([1.0 if s.params.greedy else s.params.temperature for s in seqs])
            top_k = torch.tensor([s.params.top_k for s in seqs])
            top_p = torch.tensor([s.params.top_p for s in seqs])
            probs = self._filtered_probs(logits, temp, top_k, top_p)
            E = torch.empty_like(probs)
            for i, s in enumerate(seqs):               # seeded rows draw from their own generator
                if s.generator is not None:
                    E[i].exponential_(generator=s.generator)
            if any(s.generator is None for s in seqs):
                unseeded = torch.tensor([s.generator is None for s in seqs], device=dev)
                Eg = torch.empty_like(probs).exponential_(generator=self.gen)
                E = torch.where(unseeded[:, None], Eg, E)
            sampled = (probs / E).argmax(dim=-1)
            tokens = torch.where(greedy, logits.argmax(dim=-1), sampled)
        packed = torch.stack([tokens.to(torch.int64), bad.to(torch.int64)])
        host = packed.cpu().numpy()                                      # ONE GPU->CPU copy
        return host[0], host[1].astype(bool)
