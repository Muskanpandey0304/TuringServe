"""Incremental detokenizer (ARCH 6.12).

Uses the tokenizers library's Rust ``DecodeStream`` when available; otherwise the
prefix/read-offset algorithm. Both emit text only when it does not end in an incomplete
UTF-8 character.
"""

from __future__ import annotations


class Detokenizer:
    def __init__(self, tokenizer, use_decode_stream: bool = True) -> None:
        self.tokenizer = tokenizer
        self.stream = None
        self._backend = getattr(tokenizer, "backend_tokenizer", None)
        if tokenizer is not None and use_decode_stream and self._backend is not None:
            try:
                from tokenizers.decoders import DecodeStream

                self.stream = DecodeStream(skip_special_tokens=True)
            except Exception:
                self.stream = None
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def step(self, token_id: int) -> str:
        """Feed one token; return the new text (possibly empty)."""
        if self.tokenizer is None:
            return ""
        if self.stream is not None:
            piece = self.stream.step(self._backend, token_id)
            return piece or ""
        return self._fallback(token_id)

    def _fallback(self, token_id: int) -> str:
        tok = self.tokenizer
        self.ids.append(token_id)
        prefix_text = tok.decode(self.ids[self.prefix_offset:self.read_offset],
                                 skip_special_tokens=True)
        new_text = tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            delta = new_text[len(prefix_text):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            return delta
        return ""
