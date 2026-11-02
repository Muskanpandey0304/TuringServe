"""Batch-1 decode speed vs the memory-bandwidth speed of light (ARCH 2.7, Build Guide Phase 2).

  python -m benchmarks.micro.decode_speed --model $TS_MODEL_05 [--int8] [--bandwidth 192]

tokens/s excludes the first token; speed of light = B_meas / weight bytes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--kernels", default=None)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--bandwidth", type=float, default=None, help="B_meas in GB/s (else measured)")
    a = ap.parse_args()
    from turingserve.model.loader import load_weights
    from turingserve.model.qwen2 import Qwen2Model
    from turingserve.model.qwen2_config import Qwen2Config
    from turingserve.ops import get_ops
    from turingserve.runner.contiguous_cache import ContiguousCache

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    cfg = Qwen2Config.from_dir(a.model)
    w = load_weights(a.model, cfg, dev, dtype, "int8" if a.int8 else "none")
    model = Qwen2Model(cfg, w, get_ops(a.kernels or ("triton" if dev.type == "cuda" else "reference")),
                       dev, dtype, 4096)
    weight_bytes = sum(t.numel() * t.element_size() for t in _tensors(w))
    cache = ContiguousCache(cfg, a.prompt_len + a.new_tokens + 1, dev, dtype)
    ids = torch.randint(1000, 50000, (a.prompt_len,), device=dev)
    with torch.inference_mode():
        h = model.forward_contiguous(ids, torch.arange(a.prompt_len, device=dev), cache, 0)
        tok = model.compute_logits(h, torch.tensor([a.prompt_len - 1], device=dev)).argmax(-1)
        for i in range(a.new_tokens + 8):             # first 8 steps are warm-up
            if i == 8:
                torch.cuda.synchronize() if dev.type == "cuda" else None
                t0 = time.perf_counter()
            pos = a.prompt_len + i
            h = model.forward_contiguous(tok, torch.tensor([pos], device=dev), cache, pos)
            tok = model.compute_logits(h, torch.tensor([0], device=dev)).argmax(-1)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        dt = time.perf_counter() - t0
    tps = a.new_tokens / dt
    print(f"{dev}: {tps:.1f} tok/s  (weights {weight_bytes / 1e9:.2f} GB)")
    bw = a.bandwidth
    if bw is None and dev.type == "cuda":
        from scripts.measure_bandwidth import measure
        bw = measure()
    if bw:
        sol = bw * 1e9 / weight_bytes
        print(f"B_meas {bw:.0f} GB/s -> speed of light {sol:.1f} tok/s; reached {tps / sol:.0%}")


def _tensors(w):
    yield w.embed
    yield w.w_norm
    for L in w.layers:
        yield L.w_in
        yield L.w_post
        for lw in (L.qkv, L.o, L.gu, L.down):
            yield lw.w
            if lw.b is not None:
                yield lw.b
            if lw.scale is not None:
                yield lw.scale


if __name__ == "__main__":
    main()
