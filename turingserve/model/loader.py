"""Checkpoint loading (ARCH 6.7): load on CPU, fuse, convert, check finite, move to device.

Every checkpoint tensor must be used exactly once.
"""

from __future__ import annotations

import glob
import json
import os

import torch
from safetensors import safe_open

from turingserve.model.layers import LayerWeights, LinearWeight, Weights
from turingserve.model.qwen2_config import Qwen2Config


def _tensor_files(path: str) -> dict[str, str]:
    """name -> file path."""
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        return {k: os.path.join(path, v) for k, v in wm.items()}
    out: dict[str, str] = {}
    for fn in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(fn, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = fn
    if not out:
        raise FileNotFoundError(f"no *.safetensors files in {path}")
    return out


class _Reader:
    def __init__(self, path: str) -> None:
        self.files = _tensor_files(path)
        self.used: set[str] = set()
        self._handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        if name not in self.files:
            raise KeyError(f"checkpoint is missing tensor {name!r}")
        fn = self.files[name]
        if fn not in self._handles:
            self._handles[fn] = safe_open(fn, framework="pt", device="cpu")
        self.used.add(name)
        return self._handles[fn].get_tensor(name)

    def unused(self) -> list[str]:
        return sorted(set(self.files) - self.used)


def _prep(t: torch.Tensor, dtype: torch.dtype, device, what: str) -> torch.Tensor:
    t = t.to(dtype)
    if not torch.isfinite(t).all():
        raise ValueError(f"non-finite values in {what} after conversion to {dtype}")
    return t.to(device)


def load_weights(path: str, cfg: Qwen2Config, device, dtype: torch.dtype,
                 quantization: str = "none", int8_lm_head: bool = False) -> Weights:
    from turingserve.quant.int8 import quantize

    r = _Reader(path)

    def lin(w: torch.Tensor, b: torch.Tensor | None, what: str) -> LinearWeight:
        w = w.to(dtype)
        if not torch.isfinite(w).all():
            raise ValueError(f"non-finite values in {what}")
        b = _prep(b, dtype, device, what + ".bias") if b is not None else None
        if quantization == "int8":
            wq, scale = quantize(w.float())
            return LinearWeight(wq.to(device), b, scale.to(device))
        return LinearWeight(w.to(device), b)

    embed = _prep(r.get("model.embed_tokens.weight"), dtype, device, "embed_tokens")
    layers = []
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        qkv_w = torch.cat([r.get(p + f"self_attn.{n}_proj.weight") for n in "qkv"], dim=0)
        qkv_b = torch.cat([r.get(p + f"self_attn.{n}_proj.bias") for n in "qkv"], dim=0)
        gu_w = torch.cat([r.get(p + f"mlp.{n}_proj.weight") for n in ("gate", "up")], dim=0)
        layers.append(LayerWeights(
            w_in=_prep(r.get(p + "input_layernorm.weight"), dtype, device, p + "input_layernorm"),
            qkv=lin(qkv_w, qkv_b, p + "qkv"),
            o=lin(r.get(p + "self_attn.o_proj.weight"), None, p + "o_proj"),
            w_post=_prep(r.get(p + "post_attention_layernorm.weight"), dtype, device,
                         p + "post_attention_layernorm"),
            gu=lin(gu_w, None, p + "gate_up"),
            down=lin(r.get(p + "mlp.down_proj.weight"), None, p + "down_proj")))
    w_norm = _prep(r.get("model.norm.weight"), dtype, device, "model.norm")
    unused = r.unused()
    if cfg.tie_word_embeddings and "lm_head.weight" in unused:
        unused.remove("lm_head.weight")          # tied: the file may carry a duplicate
    if unused:
        raise ValueError(f"checkpoint tensors not used: {unused[:5]}{'...' if len(unused) > 5 else ''}")
    if not cfg.tie_word_embeddings:
        raise ValueError("untied lm_head is not supported (Qwen2.5 0.5B/1.5B are tied)")
    weights = Weights(embed=embed, layers=layers, w_norm=w_norm)
    if int8_lm_head:
        wq, scale = quantize(embed.float())
        weights.embed_q = (wq, scale)
    return weights
