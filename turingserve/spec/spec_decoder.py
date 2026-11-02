"""Speculative decoder (ARCH 6.13): propose -> verify in ONE target forward -> accept/reject."""

from __future__ import annotations

import time

import torch

from turingserve.engine.sequence import BatchPlan, RequestOutput
from turingserve.sampling.rejection import accept_greedy, accept_sampling


class SpecDecoder:
    def __init__(self, engine, proposer, k: int, max_k: int, max_batch: int,
                 method: str = "ngram") -> None:
        self.engine = engine
        self.proposer = proposer
        self.method = method
        self.k = k                    # current number of drafts (Autopilot R4/R5 move it)
        self.max_k = max_k
        self.max_batch = max_batch
        self.enabled = True           # Autopilot R4 can switch speculation off for a while
        self.off_until = 0.0
        self.drafted = 0
        self.accepted = 0
        self.verify_steps = 0
        self.last_accept_mean: float | None = None    # mean accepted drafts in the last step

    # -- gating ----------------------------------------------------------------------------------
    def should_run(self, plan: BatchPlan) -> bool:
        if not self.enabled or time.monotonic() < self.off_until:
            return False
        if plan.num_decode != len(plan.items) or len(plan.items) > self.max_batch:   # R3
            return False
        return not any(it.seq.params.uses_penalties for it in plan.items)

    # -- the step ---------------------------------------------------------------------------------
    def step(self, plan: BatchPlan) -> list[RequestOutput]:
        eng = self.engine
        seqs = [it.seq for it in plan.items]
        for s in seqs:
            if s.first_scheduled_time is None:
                s.first_scheduled_time = time.perf_counter()
        ks = []
        for s in seqs:
            remaining = min(s.params.max_tokens - s.num_output,
                            eng.cfg.max_model_len - len(s.token_ids))
            ks.append(max(0, min(self.k, remaining - 1)))
        t0 = time.perf_counter()
        # the draft model needs target block tables with room for 1 + k tokens
        if self.method == "draft":
            fitted = []
            for s, k in zip(seqs, ks, strict=False):
                k = self._fit(s, k)
                eng.kv.allocate(s, 1 + k)                 # draft positions need blocks up front
                fitted.append(k)
            ks = fitted
        proposals = self.proposer.propose(seqs, ks, eng)
        entries, qs, ds = [], [], []
        for s, (drafts, q) in zip(seqs, proposals, strict=False):
            drafts = drafts[: self._fit(s, len(drafts))]
            if q is not None:
                q = q[: len(drafts)]
            ds.append(drafts)
            qs.append(q)
            eng.kv.allocate(s, 1 + len(drafts))
            entries.append((s, [s.token_ids[s.num_computed_tokens]] + drafts))
        if not any(ds):
            return eng._normal_step(plan)                     # nothing to verify
        eng.timer.t["propose"] = time.perf_counter() - t0
        eng.timer.start("forward")
        logits = eng.runner.execute_verify(entries)
        eng.timer.stop("forward")

        eng.timer.start("output")
        outputs: list[RequestOutput] = []
        step_accepted = 0
        for (s, _toks), drafts, q, lg in zip(entries, ds, qs, logits, strict=False):
            old_computed, L0, k = s.num_computed_tokens, len(s.token_ids), len(drafts)
            if not bool(torch.isfinite(lg).all()):
                outputs.append(eng._finish_with_error(s))
                continue
            rows = torch.stack([
                eng.sampler.mask_stop_tokens(lg[i], s) if s.num_output + i < s.params.min_tokens
                else lg[i] for i in range(k + 1)])
            if s.params.greedy:
                m, nxt = accept_greedy(drafts, rows)
            else:
                p = torch.stack([eng.sampler.processed_probs(lg[i], s, s.num_output + i)
                                 for i in range(k + 1)])
                m, nxt = accept_sampling(drafts, p, q, s.generator)
            self.drafted += k
            self.accepted += m
            step_accepted += m
            eng.metrics.spec_draft.inc(k)
            eng.metrics.spec_accepted.inc(m)
            s.num_computed_tokens = old_computed + 1 + m
            if self.method == "draft":
                s.draft_computed = L0 + min(m, max(k - 1, 0))
            for tok in drafts[:m] + [nxt]:
                out = eng.commit_token(s, tok)
                outputs.append(out)
                if out.finished:
                    break
            if not s.finished:
                eng.kv.trim(s)
            eng.prefix_cache.register_full_blocks(s)
        self.verify_steps += 1
        self.last_accept_mean = step_accepted / max(1, len(entries))
        eng._finalize_finished(outputs)
        eng.timer.stop("output")
        return outputs

    def _fit(self, seq, k: int) -> int:
        """Largest k' <= k whose 1 + k' tokens fit in the KV cache."""
        while k > 0 and not self.engine.kv.can_append(seq, 1 + k):
            k -= 1
        return k
