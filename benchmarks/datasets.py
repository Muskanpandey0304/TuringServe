"""Benchmark workloads (ARCH 10.2). Every loader returns BenchItem(prompt_text_with_template,
max_tokens, ignore_eos, prompt_tokens); seed 0. Prompts go to /v1/completions as TEXT with the chat
template already applied, so every engine sees the same tokens (ARCH 10.3)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass


@dataclass
class BenchItem:
    prompt: str
    max_tokens: int
    ignore_eos: bool
    prompt_tokens: int


def _templated(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _count(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def load_sharegpt(path: str, tok, n: int = 500, seed: int = 0) -> list[BenchItem]:
    """Prompt = first human turn; output length = tokens of the first assistant turn.
    Keep 4 <= prompt <= 1024, output >= 4, prompt + output <= 2048."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for conv in data:
        t = conv.get("conversations", [])
        if len(t) < 2 or t[0].get("from") != "human" or t[1].get("from") != "gpt":
            continue
        prompt = _templated(tok, [{"role": "user", "content": t[0]["value"]}])
        p = _count(tok, prompt)
        o = _count(tok, t[1]["value"])
        if 4 <= p <= 1024 and o >= 4 and p + o <= 2048:
            items.append(BenchItem(prompt, o, True, p))
    random.Random(seed).shuffle(items)
    return items[:n]


def _wikitext_ids(tok):
    from datasets import load_dataset

    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return tok("\n\n".join(wt["text"]), add_special_tokens=False).input_ids


def _cut_text(tok, ids: list[int], target_total: int) -> BenchItem:
    """Cut raw text so the TEMPLATED prompt has (about) ``target_total`` tokens."""
    overhead = _count(tok, _templated(tok, [{"role": "user", "content": ""}]))
    n = max(1, target_total - overhead)
    for _ in range(4):
        prompt = _templated(tok, [{"role": "user", "content": tok.decode(ids[:n])}])
        p = _count(tok, prompt)
        if p == target_total:
            break
        n += target_total - p
    return BenchItem(prompt, 1, True, p)


def load_synthetic(tok, n: int = 200, prompt_len: int = 512, out_len: int = 128,
                   seed: int = 0) -> list[BenchItem]:
    ids = _wikitext_ids(tok)
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        start = rng.randrange(0, len(ids) - prompt_len - 8)
        it = _cut_text(tok, ids[start:start + prompt_len + 8], prompt_len)
        it.max_tokens = out_len
        items.append(it)
    return items


def load_prefill(tok, lengths=(2048, 4096, 8000), n_each: int = 5, seed: int = 0
                 ) -> list[BenchItem]:
    ids = _wikitext_ids(tok)
    rng = random.Random(seed)
    items = []
    for L in lengths:
        for _ in range(n_each):
            start = rng.randrange(0, len(ids) - L - 64)
            items.append(_cut_text(tok, ids[start:start + L + 64], L))   # max_tokens = 1
    return items


def load(name: str, tok, n: int, args=None) -> list[BenchItem]:
    import os

    if name == "sharegpt":
        return load_sharegpt(os.environ["TS_SHAREGPT"], tok, n)
    if name == "synthetic":
        return load_synthetic(tok, n)
    if name == "prefill":
        return load_prefill(tok)
    raise ValueError(name)
