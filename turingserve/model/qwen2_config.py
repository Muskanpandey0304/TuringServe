"""Qwen2 configuration (ARCH 6.7, 2.6)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen2Config:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    group_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]
    bos_token_id: int | None

    @property
    def qkv_dim(self) -> int:
        return (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def kv_bytes_per_token(self) -> int:
        """2 (K and V) x layers x KVH x D x 2 bytes (fp16)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2

    @classmethod
    def from_dict(cls, c: dict, gen: dict | None = None) -> Qwen2Config:
        if c.get("model_type") != "qwen2":
            raise ValueError(f"unsupported model_type {c.get('model_type')!r}: only 'qwen2' is supported")
        if c.get("use_sliding_window"):
            raise ValueError("use_sliding_window=true is not supported")
        if c.get("rope_scaling"):
            raise ValueError("rope_scaling is not supported")
        rp = c.get("rope_parameters") or {}                       # transformers >= 5 layout
        if rp.get("rope_type", "default") != "default":
            raise ValueError(f"rope_type {rp['rope_type']!r} is not supported")
        theta = c.get("rope_theta", rp.get("rope_theta", 10000.0))
        eos = (gen or {}).get("eos_token_id", c.get("eos_token_id"))
        if eos is None:
            eos = ()
        eos_ids = (int(eos),) if isinstance(eos, int) else tuple(int(x) for x in eos)
        heads, kv = c["num_attention_heads"], c["num_key_value_heads"]
        if heads % kv:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        return cls(
            hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"], num_heads=heads, num_kv_heads=kv,
            head_dim=c["hidden_size"] // heads, group_size=heads // kv, vocab_size=c["vocab_size"],
            rms_norm_eps=float(c.get("rms_norm_eps", 1e-6)),
            rope_theta=float(theta),
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
            eos_token_ids=eos_ids, bos_token_id=c.get("bos_token_id"))

    @classmethod
    def from_dir(cls, path: str) -> Qwen2Config:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
        gen = None
        gp = os.path.join(path, "generation_config.json")
        if os.path.exists(gp):
            with open(gp, encoding="utf-8") as f:
                gen = json.load(f)
        return cls.from_dict(c, gen)
