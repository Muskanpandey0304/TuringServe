"""Shared helpers for the correctness tests."""

from __future__ import annotations

import torch

from turingserve.model.loader import load_weights
from turingserve.model.qwen2 import Qwen2Model
from turingserve.model.qwen2_config import Qwen2Config
from turingserve.ops import get_ops


def load_ours(path, device="cpu", dtype=torch.float32, kernels="reference", quant="none"):
    cfg = Qwen2Config.from_dir(path)
    w = load_weights(path, cfg, torch.device(device), dtype, quant)
    return Qwen2Model(cfg, w, get_ops(kernels), torch.device(device), dtype, 4096), cfg


def load_hf(path, dtype=torch.float32, device="cpu"):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    return m


def rand_ids(n, seed=0, lo=100, hi=50000):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi, (n,), generator=g).tolist()


def hf_greedy(hf, prompt, n, eos):
    with torch.inference_mode():
        out = hf.generate(torch.tensor([prompt]), do_sample=False, repetition_penalty=1.0,
                          temperature=None, top_p=None, top_k=None, max_new_tokens=n,
                          eos_token_id=list(eos), pad_token_id=151643)
    return out[0, len(prompt):].tolist()
