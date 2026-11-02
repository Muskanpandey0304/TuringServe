"""WikiText-2 perplexity (Build Guide 0.11, ARCH 10.6). Window 1024, stride 512; the loss is
counted only on the new tokens of each window; ppl = exp(sum nll / sum tokens).

  python scripts/perplexity.py --model $TS_MODEL_05 --engine hf --dtype float16
  python scripts/perplexity.py --model $TS_MODEL_05 --engine turingserve --dtype float16 [--int8]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tokens(model_path: str) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(data["text"])
    return AutoTokenizer.from_pretrained(model_path)(text, return_tensors="pt").input_ids[0]


def nll_sum(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 256) -> tuple[float, int]:
    """logits [n, V] for positions 0..n-1 vs targets [n] (-100 ignored), in chunks of 256."""
    total, count = 0.0, 0
    for i in range(0, logits.shape[0], chunk):
        lg = logits[i:i + chunk].float()
        tg = targets[i:i + chunk]
        m = tg != -100
        if m.any():
            total += torch.nn.functional.cross_entropy(lg[m], tg[m], reduction="sum").item()
            count += int(m.sum())
    return total, count


def get_logits_fn(args):
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    if args.engine == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).cuda().eval()
        return lambda ids: model(ids[None].cuda()).logits[0]
    from turingserve.model.loader import load_weights
    from turingserve.model.qwen2 import Qwen2Model
    from turingserve.model.qwen2_config import Qwen2Config
    from turingserve.ops import get_ops

    cfg = Qwen2Config.from_dir(args.model)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernels = args.kernels or ("triton" if dev.type == "cuda" else "reference")
    w = load_weights(args.model, cfg, dev, dtype, "int8" if args.int8 else "none")
    model = Qwen2Model(cfg, w, get_ops(kernels), dev, dtype, 4096)

    def fn(ids):
        ids = ids.to(dev)
        h = model.forward_dense(ids, torch.arange(len(ids), device=dev))
        return model.compute_logits(h, torch.arange(len(ids), device=dev))
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--engine", default="hf", choices=["hf", "turingserve"])
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--kernels", default=None)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ids = load_tokens(args.model)
    fn = get_logits_fn(args)
    N = len(ids)
    prev_end, total, count, windows = 0, 0.0, 0, 0
    with torch.inference_mode():
        for begin in range(0, N, args.stride):
            end = min(begin + args.window, N)
            trg_len = end - prev_end
            chunk = ids[begin:end]
            target = chunk.clone()
            target[:-trg_len] = -100
            logits = fn(chunk)
            # logits[:-1] predict target[1:]
            t, c = nll_sum(logits[:-1], target[1:].to(logits.device))
            total, count = total + t, count + c
            prev_end = end
            windows += 1
            if end == N or (args.max_windows and windows >= args.max_windows):
                break
    ppl = math.exp(total / count)
    print(f"engine={args.engine} dtype={args.dtype} int8={args.int8} tokens={count} ppl={ppl:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"engine": args.engine, "dtype": args.dtype, "int8": args.int8,
                       "model": args.model, "tokens": count, "ppl": ppl}, f, indent=1)


if __name__ == "__main__":
    main()
