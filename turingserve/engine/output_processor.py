"""Turns sampled tokens into text deltas and finish decisions (ARCH 6.12)."""

from __future__ import annotations

from turingserve.engine.detokenizer import Detokenizer
from turingserve.engine.sequence import RequestOutput, Sequence, SeqStatus
from turingserve.engine.stop import find_stop_string, hold_back, is_stop_token


class OutputProcessor:
    def __init__(self, tokenizer, eos_ids: tuple[int, ...], max_model_len: int) -> None:
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.max_model_len = max_model_len

    def new_detokenizer(self) -> Detokenizer:
        return Detokenizer(self.tokenizer)

    def make_output(self, seq: Sequence, delta: str, tokens: list[int]) -> RequestOutput:
        return RequestOutput(
            request_id=seq.request_id, delta_text=delta, delta_token_ids=tokens,
            finished=seq.status == SeqStatus.FINISHED, finish_reason=seq.finish_reason,
            prompt_tokens=seq.num_prompt_tokens, completion_tokens=seq.num_output,
            cached_prompt_tokens=seq.num_cached_prompt_tokens)

    def process(self, seq: Sequence, token: int) -> RequestOutput:
        """``token`` is already appended to ``seq.token_ids``; returns the output for it."""
        # 1. EOS / stop token: finish; the token's text is not emitted
        if is_stop_token(seq, token, self.eos_ids):
            return self._finish(seq, "stop", [token])
        # 2. detokenize, then search the stop strings
        piece = seq.detok.step(token) if seq.detok is not None else ""
        seq.text += piece
        if seq.params.stop:
            k = find_stop_string(seq)
            if k != -1:
                k = max(k, seq.emitted_len)
                delta = seq.text[seq.emitted_len:k]
                seq.text = seq.text[:k]
                seq.emitted_len = k
                return self._finish(seq, "stop", [token], delta=delta, flush=False)
        # 3. length
        if seq.num_output >= seq.params.max_tokens or len(seq.token_ids) >= self.max_model_len:
            return self._finish(seq, "length", [token])
        # 4. emit, holding back what could be the start of a stop string
        upto = len(seq.text) - hold_back(seq)
        delta = seq.text[seq.emitted_len:upto] if upto > seq.emitted_len else ""
        seq.emitted_len += len(delta)
        return self.make_output(seq, delta, [token])

    def _finish(self, seq: Sequence, reason: str, tokens: list[int], delta: str | None = None,
                flush: bool = True) -> RequestOutput:
        if flush:
            delta = seq.text[seq.emitted_len:]
            seq.emitted_len = len(seq.text)
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        return self.make_output(seq, delta or "", tokens)
