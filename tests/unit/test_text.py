"""Detokenizer, stop strings and finish reasons (Phase 5)."""

import pytest

from turingserve.engine.output_processor import OutputProcessor
from turingserve.engine.sampling_params import SamplingParams
from turingserve.engine.sequence import Sequence

TEXTS = ["The quick brown fox jumps over the lazy dog.",
         "नमस्ते दुनिया, यह एक परीक्षण है", "🙂👍🏽🚀", "你好世界",
         "def f(x):\n\treturn x * 2\n\n\tprint('hi')"]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("stream", [True, False])
def test_detokenizer_matches_decode(tokenizer, text, stream):
    from turingserve.engine.detokenizer import Detokenizer

    ids = tokenizer.encode(text)
    d = Detokenizer(tokenizer, use_decode_stream=stream)
    out = []
    for t in ids:
        piece = d.step(t)
        assert "�" not in piece
        out.append(piece)
    assert "".join(out) == tokenizer.decode(ids, skip_special_tokens=True)


class CharTok:
    """One token per character, for exact stop-string tests."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def run(text, params, tok=None, eos=(0,), max_len=10_000, ids=None):
    """Feed ``text`` char by char through the OutputProcessor; returns (emitted, finish_reason)."""
    op = OutputProcessor(CharTok(), eos, max_len)
    seq = Sequence("r", [65, 66], 2, params, detok=op.new_detokenizer())
    emitted, reason = "", None
    deltas = []
    for ch in (ids if ids is not None else [ord(c) for c in text]):
        seq.token_ids.append(ch)
        out = op.process(seq, ch)
        emitted += out.delta_text
        deltas.append(out.delta_text)
        if out.finished:
            reason = out.finish_reason
            break
    return emitted, reason, deltas, seq


def test_stop_string_truncates_and_finishes():
    e, r, *_ = run("hello STOP world", SamplingParams(stop=["STOP"], max_tokens=100))
    assert e == "hello " and r == "stop"


def test_stop_string_never_emitted_early():
    _, _, deltas, _ = run("abcSTOPdef", SamplingParams(stop=["STOP"], max_tokens=100))
    assert all("S" not in d or d == "" for d in deltas[:-1]) or "".join(deltas) == "abc"
    assert "".join(deltas) == "abc"


def test_stop_at_first_token():
    e, r, *_ = run("STOPx", SamplingParams(stop=["STOP"], max_tokens=100))
    assert e == "" and r == "stop"


def test_earliest_of_several_stops_wins():
    e, r, *_ = run("aa-two-one-", SamplingParams(stop=["one", "two"], max_tokens=100))
    assert e == "aa-" and r == "stop"


def test_partial_stop_prefix_flushed_at_length():
    e, r, *_ = run("abST", SamplingParams(stop=["STOP"], max_tokens=4))
    assert e == "abST" and r == "length"


def test_finish_reasons():
    e, r, *_ = run("", SamplingParams(max_tokens=100), ids=[104, 105, 0])
    assert (e, r) == ("hi", "stop")                            # EOS
    _, r, *_ = run("abcdef", SamplingParams(max_tokens=3))
    assert r == "length"
    _, r, *_ = run("abcdef", SamplingParams(max_tokens=100), max_len=5)
    assert r == "length"                                       # max_model_len


def test_ignore_eos_continues():
    _, r, _, seq = run("", SamplingParams(max_tokens=4, ignore_eos=True), ids=[0, 0, 0, 0, 0])
    assert r == "length" and seq.num_output == 4


def test_sampling_params_validation_names_field():
    for kw, field in [({"temperature": 3}, "temperature"), ({"top_p": 0}, "top_p"),
                      ({"top_k": 0}, "top_k"), ({"stop": [""]}, "stop"),
                      ({"stop": ["a"] * 17}, "stop"), ({"presence_penalty": 3}, "presence_penalty")]:
        with pytest.raises(ValueError, match=field):
            SamplingParams(**kw)
